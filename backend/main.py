#!/usr/bin/env python3
"""
PTFlow - PT刷流管理系统后端
支持多qBittorrent实例、动态限速、RSS订阅
"""

import os
import sys
import json
import time
import asyncio
import hashlib
import secrets
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
from functools import reduce
import logging
import sqlite3
import re

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
import jwt
import qbittorrentapi
from qbittorrentapi.exceptions import APIConnectionError, LoginFailed
import feedparser
import requests

# ════════════════════════════════════════════════════════════════════════════════
# 配置与常量
# ════════════════════════════════════════════════════════════════════════════════

VERSION = "1.0.0"
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
DATABASE_PATH = os.getenv("DATABASE_PATH", "ptflow.db")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ptflow")

# 动态限速常量 (来自原脚本)
class LimitConst:
    PHASE_WARMUP = "warmup"
    PHASE_CATCH = "catch"
    PHASE_STEADY = "steady"
    PHASE_FINISH = "finish"
    
    FINISH_TIME = 30
    STEADY_TIME = 120
    
    PRECISION_PERFECT = 0.001
    PRECISION_GOOD = 0.005
    
    SPEED_PROTECT_RATIO = 2.5
    SPEED_PROTECT_LIMIT = 1.3
    PROGRESS_PROTECT = 0.90
    
    MIN_LIMIT = 4096
    MAX_REANNOUNCE = 86400
    
    PID_PARAMS = {
        'warmup': {'kp': 0.3, 'ki': 0.05, 'kd': 0.02, 'headroom': 1.03},
        'catch':  {'kp': 0.5, 'ki': 0.10, 'kd': 0.05, 'headroom': 1.02},
        'steady': {'kp': 0.6, 'ki': 0.15, 'kd': 0.08, 'headroom': 1.005},
        'finish': {'kp': 0.8, 'ki': 0.20, 'kd': 0.12, 'headroom': 1.001},
    }
    
    QUANT_STEPS = {'finish': 256, 'steady': 512, 'catch': 2048, 'warmup': 4096}
    
    KALMAN_Q_SPEED = 0.1
    KALMAN_Q_ACCEL = 0.05
    KALMAN_R = 0.5
    
    SPEED_WINDOWS = [5, 15, 30, 60]
    WINDOW_WEIGHTS = {
        'warmup': {5: 0.1, 15: 0.2, 30: 0.3, 60: 0.4},
        'catch':  {5: 0.2, 15: 0.3, 30: 0.3, 60: 0.2},
        'steady': {5: 0.3, 15: 0.3, 30: 0.2, 60: 0.2},
        'finish': {5: 0.5, 15: 0.3, 30: 0.15, 60: 0.05},
    }
    
    ANNOUNCE_INTERVAL_NEW = 1800
    ANNOUNCE_INTERVAL_WEEK = 2700
    ANNOUNCE_INTERVAL_OLD = 3600


# ════════════════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════════════════

def fmt_size(b: float, precision: int = 2) -> str:
    if b == 0: return "0 B"
    for u in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(b) < 1024: return f"{b:.{precision}f} {u}"
        b /= 1024
    return f"{b:.{precision}f} PiB"

def fmt_speed(b: float, precision: int = 1) -> str:
    if b == 0: return "0 B/s"
    for u in ['B/s', 'KiB/s', 'MiB/s', 'GiB/s']:
        if abs(b) < 1024: return f"{b:.{precision}f} {u}"
        b /= 1024
    return f"{b:.{precision}f} TiB/s"

def fmt_duration(s: float) -> str:
    s = max(0, int(s))
    if s < 60: return f"{s}s"
    if s < 3600: return f"{s//60}m{s%60}s"
    return f"{s//3600}h{(s%3600)//60}m"

def safe_div(a: float, b: float, default: float = 0) -> float:
    try:
        if b == 0 or abs(b) < 1e-10: return default
        return a / b
    except: return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))

def get_phase(tl: float, synced: bool) -> str:
    if not synced: return LimitConst.PHASE_WARMUP
    if tl <= LimitConst.FINISH_TIME: return LimitConst.PHASE_FINISH
    if tl <= LimitConst.STEADY_TIME: return LimitConst.PHASE_STEADY
    return LimitConst.PHASE_CATCH

def estimate_announce_interval(time_ref: float) -> int:
    age = time.time() - time_ref
    if age < 7 * 86400: return LimitConst.ANNOUNCE_INTERVAL_NEW
    elif age < 30 * 86400: return LimitConst.ANNOUNCE_INTERVAL_WEEK
    return LimitConst.ANNOUNCE_INTERVAL_OLD


# ════════════════════════════════════════════════════════════════════════════════
# 动态限速控制器
# ════════════════════════════════════════════════════════════════════════════════

class PIDController:
    def __init__(self):
        self.kp = 0.6
        self.ki = 0.15
        self.kd = 0.08
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = 0.0
        self._last_output = 1.0
        self._initialized = False
        self._integral_limit = 0.3
        self._derivative_filter = 0.0
    
    def set_phase(self, phase: str):
        params = LimitConst.PID_PARAMS.get(phase, LimitConst.PID_PARAMS['steady'])
        self.kp, self.ki, self.kd = params['kp'], params['ki'], params['kd']
    
    def update(self, setpoint: float, measured: float, now: float) -> float:
        error = safe_div(setpoint - measured, max(setpoint, 1), 0)
        if not self._initialized:
            self._last_error = error
            self._last_time = now
            self._initialized = True
            return 1.0
        dt = now - self._last_time
        if dt <= 0.01: return self._last_output
        self._last_time = now
        
        p_term = self.kp * error
        self._integral = clamp(self._integral + error * dt, -self._integral_limit, self._integral_limit)
        i_term = self.ki * self._integral
        
        raw_derivative = (error - self._last_error) / dt
        self._derivative_filter = 0.3 * raw_derivative + 0.7 * self._derivative_filter
        d_term = self.kd * self._derivative_filter
        self._last_error = error
        
        output = clamp(1.0 + p_term + i_term + d_term, 0.5, 2.0)
        self._last_output = output
        return output
    
    def reset(self):
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = 0.0
        self._last_output = 1.0
        self._derivative_filter = 0.0
        self._initialized = False


class ExtendedKalman:
    def __init__(self):
        self.speed = 0.0
        self.accel = 0.0
        self.p00 = 1000.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 1000.0
        self._last_time = 0.0
        self._initialized = False
    
    def update(self, measurement: float, now: float) -> Tuple[float, float]:
        if not self._initialized:
            self.speed = measurement
            self._last_time = now
            self._initialized = True
            return measurement, 0.0
        dt = now - self._last_time
        if dt <= 0.01: return self.speed, self.accel
        self._last_time = now
        
        pred_speed = self.speed + self.accel * dt
        p00_pred = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11 + LimitConst.KALMAN_Q_SPEED
        p01_pred = self.p01 + dt * self.p11
        p10_pred = self.p10 + dt * self.p11
        p11_pred = self.p11 + LimitConst.KALMAN_Q_ACCEL
        
        s = p00_pred + LimitConst.KALMAN_R
        if abs(s) < 1e-10: return self.speed, self.accel
        k0, k1 = p00_pred / s, p10_pred / s
        innovation = measurement - pred_speed
        
        self.speed = pred_speed + k0 * innovation
        self.accel = self.accel + k1 * innovation
        self.p00 = (1 - k0) * p00_pred
        self.p01 = (1 - k0) * p01_pred
        self.p10 = -k1 * p00_pred + p10_pred
        self.p11 = -k1 * p01_pred + p11_pred
        return self.speed, self.accel
    
    def predict_upload(self, seconds: float) -> float:
        return max(0, self.speed * seconds + 0.5 * self.accel * seconds * seconds)
    
    def reset(self):
        self.speed = 0.0
        self.accel = 0.0
        self.p00 = 1000.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 1000.0
        self._initialized = False


class MultiWindowSpeedTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._samples: deque = deque(maxlen=1200)
    
    def record(self, now: float, speed: float):
        with self._lock:
            self._samples.append((now, speed))
    
    def get_weighted_avg(self, now: float, phase: str) -> float:
        weights = LimitConst.WINDOW_WEIGHTS.get(phase, LimitConst.WINDOW_WEIGHTS['steady'])
        with self._lock:
            samples = list(self._samples)
        
        total_weight = 0.0
        weighted_sum = 0.0
        for window in LimitConst.SPEED_WINDOWS:
            win_samples = [s for t, s in samples if now - t <= window]
            if win_samples:
                avg = sum(win_samples) / len(win_samples)
                w = weights.get(window, 0.25)
                weighted_sum += avg * w
                total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_recent_trend(self, now: float, window: int = 10) -> float:
        with self._lock:
            samples = [(t, s) for t, s in self._samples if now - t <= window]
        if len(samples) < 5: return 0.0
        mid = len(samples) // 2
        first = sum(s for _, s in samples[:mid]) / mid
        second = sum(s for _, s in samples[mid:]) / (len(samples) - mid)
        return safe_div(second - first, first, 0)
    
    def clear(self):
        with self._lock:
            self._samples.clear()


class AdaptiveQuantizer:
    @staticmethod
    def quantize(limit: int, phase: str, current_speed: float, target: float, trend: float = 0) -> int:
        if limit <= 0: return limit
        base = LimitConst.QUANT_STEPS.get(phase, 1024)
        ratio = safe_div(current_speed, target, 1)
        
        if phase == 'finish':
            step = 256
        elif ratio > 1.2:
            step = base * 2
        elif ratio > 1.05:
            step = base
        elif ratio > 0.8:
            step = base // 2
        else:
            step = base
        
        if abs(trend) > 0.1:
            step = max(256, step // 2)
        step = int(clamp(step, 256, 8192))
        return max(LimitConst.MIN_LIMIT, int((limit + step // 2) // step) * step)


class DynamicLimitController:
    """动态限速控制器 - 完整实现"""
    
    def __init__(self):
        self.kalman = ExtendedKalman()
        self.speed_tracker = MultiWindowSpeedTracker()
        self.pid = PIDController()
        self._smooth_limit = -1
    
    def record_speed(self, now: float, speed: float):
        self.kalman.update(speed, now)
        self.speed_tracker.record(now, speed)
    
    def calculate(self, target: float, uploaded: int, time_left: float, elapsed: float,
                  phase: str, now: float, precision_adj: float = 1.0) -> Tuple[int, str, Dict]:
        debug: Dict[str, Any] = {}
        adjusted_target = target * precision_adj
        
        kalman_speed = self.kalman.speed
        weighted_speed = self.speed_tracker.get_weighted_avg(now, phase)
        trend = self.speed_tracker.get_recent_trend(now)
        current_speed = weighted_speed if (phase == 'finish' and weighted_speed > 0) else (kalman_speed if kalman_speed > 0 else weighted_speed)
        
        total_time = elapsed + time_left
        target_total = adjusted_target * total_time
        debug['predicted_ratio'] = safe_div(uploaded + self.kalman.predict_upload(time_left), target_total, 0)
        
        need = max(0, target_total - uploaded)
        if time_left <= 0:
            return -1, "汇报中", debug
        required_speed = need / time_left
        debug['required_speed'] = required_speed
        
        self.pid.set_phase(phase)
        pid_output = self.pid.update(target_total, uploaded, now)
        debug['pid_output'] = pid_output
        
        headroom = LimitConst.PID_PARAMS.get(phase, {}).get('headroom', 1.01)
        limit = -1
        reason = ""
        
        if phase == 'finish':
            pred = debug['predicted_ratio']
            correction = max(0.8, 1 - (pred - 1) * 3) if pred > 1.002 else (min(1.2, 1 + (1 - pred) * 3) if pred < 0.998 else 1.0)
            limit = int(required_speed * pid_output * correction)
            reason = f"F:{required_speed/1024:.0f}K"
        elif phase == 'steady':
            if debug['predicted_ratio'] > 1.01:
                headroom = 1.0
            limit = int(required_speed * headroom * pid_output)
            reason = f"S:{required_speed/1024:.0f}K"
        elif phase == 'catch':
            if required_speed > adjusted_target * 5:
                limit = -1
                reason = "C:欠速放开"
            else:
                limit = int(required_speed * headroom * pid_output)
                reason = f"C:{required_speed/1024:.0f}K"
        else:  # warmup
            progress = safe_div(uploaded, target_total, 0)
            if progress >= 1.0:
                limit = LimitConst.MIN_LIMIT
                reason = f"W:超{(progress-1)*100:.0f}%"
            elif progress >= 0.8:
                limit = int(required_speed * 1.01 * pid_output)
                reason = f"W:精控"
            elif progress >= 0.5:
                limit = int(required_speed * 1.05)
                reason = f"W:温控"
            else:
                limit = -1
                reason = "W:预热"
        
        if limit > 0:
            limit = AdaptiveQuantizer.quantize(limit, phase, current_speed, adjusted_target, trend)
        limit = self._smooth(limit, phase)
        debug['final_limit'] = limit
        return limit, reason, debug
    
    def _smooth(self, new_limit: int, phase: str) -> int:
        if new_limit <= 0 or self._smooth_limit <= 0 or phase == 'finish':
            self._smooth_limit = new_limit
            return new_limit
        change = abs(new_limit - self._smooth_limit) / self._smooth_limit
        if change < 0.2:
            self._smooth_limit = new_limit
        else:
            factor = 0.5 if change >= 0.5 else 0.3
            self._smooth_limit = int((1 - factor) * self._smooth_limit + factor * new_limit)
        return self._smooth_limit
    
    def reset(self):
        self.kalman.reset()
        self.speed_tracker.clear()
        self.pid.reset()
        self._smooth_limit = -1


# ════════════════════════════════════════════════════════════════════════════════
# 数据库模型
# ════════════════════════════════════════════════════════════════════════════════

class Database:
    def __init__(self, path: str = DATABASE_PATH):
        self.path = path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        
        # 用户表
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at REAL,
            is_admin INTEGER DEFAULT 0
        )''')
        
        # qBittorrent实例表
        c.execute('''CREATE TABLE IF NOT EXISTS qb_instances (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            host TEXT NOT NULL,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            created_at REAL
        )''')
        
        # 站点配置表
        c.execute('''CREATE TABLE IF NOT EXISTS sites (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            tracker_keyword TEXT NOT NULL,
            target_speed_kib INTEGER DEFAULT 51200,
            safety_margin REAL DEFAULT 0.98,
            enable_limit INTEGER DEFAULT 1,
            enable_dl_limit INTEGER DEFAULT 1,
            enable_reannounce INTEGER DEFAULT 1,
            cookie TEXT,
            created_at REAL
        )''')
        
        # RSS订阅表
        c.execute('''CREATE TABLE IF NOT EXISTS rss_feeds (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            site_id INTEGER,
            qb_instance_id INTEGER,
            save_path TEXT,
            category TEXT,
            auto_download INTEGER DEFAULT 1,
            filter_include TEXT,
            filter_exclude TEXT,
            last_check REAL,
            created_at REAL,
            FOREIGN KEY (site_id) REFERENCES sites(id),
            FOREIGN KEY (qb_instance_id) REFERENCES qb_instances(id)
        )''')

        # RSS任务表
        c.execute('''CREATE TABLE IF NOT EXISTS rss_tasks (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            qb_instance_id INTEGER NOT NULL,
            save_path TEXT,
            category TEXT,
            tags TEXT,
            paused INTEGER DEFAULT 0,
            interval_minutes INTEGER DEFAULT 10,
            enabled INTEGER DEFAULT 1,
            last_run REAL,
            created_at REAL,
            FOREIGN KEY (qb_instance_id) REFERENCES qb_instances(id)
        )''')

        # RSS规则表
        c.execute('''CREATE TABLE IF NOT EXISTS rss_task_rules (
            id INTEGER PRIMARY KEY,
            task_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            include_keywords TEXT,
            exclude_keywords TEXT,
            save_path TEXT,
            category TEXT,
            tags TEXT,
            paused INTEGER DEFAULT 0,
            created_at REAL,
            FOREIGN KEY (task_id) REFERENCES rss_tasks(id)
        )''')

        # RSS任务下载历史
        c.execute('''CREATE TABLE IF NOT EXISTS rss_task_history (
            id INTEGER PRIMARY KEY,
            task_id INTEGER NOT NULL,
            rule_id INTEGER,
            entry_hash TEXT,
            name TEXT,
            link TEXT,
            downloaded_at REAL,
            FOREIGN KEY (task_id) REFERENCES rss_tasks(id),
            FOREIGN KEY (rule_id) REFERENCES rss_task_rules(id)
        )''')

        # 删种任务表
        c.execute('''CREATE TABLE IF NOT EXISTS delete_tasks (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            qb_instance_id INTEGER NOT NULL,
            interval_minutes INTEGER DEFAULT 60,
            enabled INTEGER DEFAULT 1,
            last_run REAL,
            created_at REAL,
            FOREIGN KEY (qb_instance_id) REFERENCES qb_instances(id)
        )''')

        # 删种规则表
        c.execute('''CREATE TABLE IF NOT EXISTS delete_task_rules (
            id INTEGER PRIMARY KEY,
            task_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            min_ratio REAL,
            min_seeding_hours REAL,
            min_uploaded_gb REAL,
            max_size_gb REAL,
            include_categories TEXT,
            exclude_categories TEXT,
            include_tags TEXT,
            exclude_tags TEXT,
            delete_files INTEGER DEFAULT 0,
            created_at REAL,
            FOREIGN KEY (task_id) REFERENCES delete_tasks(id)
        )''')
        
        # 种子状态表
        c.execute('''CREATE TABLE IF NOT EXISTS torrent_states (
            hash TEXT PRIMARY KEY,
            qb_instance_id INTEGER,
            site_id INTEGER,
            name TEXT,
            cycle_index INTEGER DEFAULT 0,
            cycle_start REAL,
            cycle_start_uploaded INTEGER,
            cycle_synced INTEGER DEFAULT 0,
            total_uploaded_start INTEGER,
            session_start_time REAL,
            last_announce_time REAL,
            updated_at REAL,
            FOREIGN KEY (qb_instance_id) REFERENCES qb_instances(id),
            FOREIGN KEY (site_id) REFERENCES sites(id)
        )''')
        
        # 统计表
        c.execute('''CREATE TABLE IF NOT EXISTS stats (
            id INTEGER PRIMARY KEY,
            qb_instance_id INTEGER,
            site_id INTEGER,
            total_cycles INTEGER DEFAULT 0,
            success_cycles INTEGER DEFAULT 0,
            precision_cycles INTEGER DEFAULT 0,
            total_uploaded INTEGER DEFAULT 0,
            start_time REAL,
            FOREIGN KEY (qb_instance_id) REFERENCES qb_instances(id),
            FOREIGN KEY (site_id) REFERENCES sites(id)
        )''')
        
        # 下载历史表
        c.execute('''CREATE TABLE IF NOT EXISTS download_history (
            id INTEGER PRIMARY KEY,
            hash TEXT,
            name TEXT,
            rss_feed_id INTEGER,
            qb_instance_id INTEGER,
            downloaded_at REAL,
            FOREIGN KEY (rss_feed_id) REFERENCES rss_feeds(id),
            FOREIGN KEY (qb_instance_id) REFERENCES qb_instances(id)
        )''')
        
        # 创建默认管理员
        c.execute('SELECT COUNT(*) FROM users')
        if c.fetchone()[0] == 0:
            password_hash = hashlib.sha256("admin".encode()).hexdigest()
            c.execute('INSERT INTO users (username, password_hash, created_at, is_admin) VALUES (?, ?, ?, ?)',
                     ('admin', password_hash, time.time(), 1))
        
        conn.commit()
        conn.close()
    
    def get_conn(self):
        return sqlite3.connect(self.path)


db = Database()


# ════════════════════════════════════════════════════════════════════════════════
# Pydantic 模型
# ════════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str

class QBInstanceCreate(BaseModel):
    name: str
    host: str
    username: str
    password: str

class QBInstanceUpdate(BaseModel):
    name: Optional[str] = None
    host: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    enabled: Optional[bool] = None

class SiteCreate(BaseModel):
    name: str
    tracker_keyword: str
    target_speed_kib: int = 51200
    safety_margin: float = 0.98
    enable_limit: bool = True
    enable_dl_limit: bool = True
    enable_reannounce: bool = True
    cookie: Optional[str] = None

class SiteUpdate(BaseModel):
    name: Optional[str] = None
    tracker_keyword: Optional[str] = None
    target_speed_kib: Optional[int] = None
    safety_margin: Optional[float] = None
    enable_limit: Optional[bool] = None
    enable_dl_limit: Optional[bool] = None
    enable_reannounce: Optional[bool] = None
    cookie: Optional[str] = None

class RSSFeedCreate(BaseModel):
    name: str
    url: str
    site_id: Optional[int] = None
    qb_instance_id: int
    save_path: Optional[str] = None
    category: Optional[str] = None
    auto_download: bool = True
    filter_include: Optional[str] = None
    filter_exclude: Optional[str] = None

class RSSFeedUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    site_id: Optional[int] = None
    qb_instance_id: Optional[int] = None
    save_path: Optional[str] = None
    category: Optional[str] = None
    auto_download: Optional[bool] = None
    filter_include: Optional[str] = None
    filter_exclude: Optional[str] = None

class RSSTaskCreate(BaseModel):
    name: str
    url: str
    qb_instance_id: int
    save_path: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[str] = None
    paused: bool = False
    interval_minutes: int = 10
    enabled: bool = True

class RSSTaskUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    qb_instance_id: Optional[int] = None
    save_path: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[str] = None
    paused: Optional[bool] = None
    interval_minutes: Optional[int] = None
    enabled: Optional[bool] = None

class RSSTaskRuleCreate(BaseModel):
    name: str
    include_keywords: Optional[str] = None
    exclude_keywords: Optional[str] = None
    save_path: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[str] = None
    paused: bool = False

class RSSTaskRuleUpdate(BaseModel):
    name: Optional[str] = None
    include_keywords: Optional[str] = None
    exclude_keywords: Optional[str] = None
    save_path: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[str] = None
    paused: Optional[bool] = None

class DeleteTaskCreate(BaseModel):
    name: str
    qb_instance_id: int
    interval_minutes: int = 60
    enabled: bool = True

class DeleteTaskUpdate(BaseModel):
    name: Optional[str] = None
    qb_instance_id: Optional[int] = None
    interval_minutes: Optional[int] = None
    enabled: Optional[bool] = None

class DeleteTaskRuleCreate(BaseModel):
    name: str
    min_ratio: Optional[float] = None
    min_seeding_hours: Optional[float] = None
    min_uploaded_gb: Optional[float] = None
    max_size_gb: Optional[float] = None
    include_categories: Optional[str] = None
    exclude_categories: Optional[str] = None
    include_tags: Optional[str] = None
    exclude_tags: Optional[str] = None
    delete_files: bool = False

class DeleteTaskRuleUpdate(BaseModel):
    name: Optional[str] = None
    min_ratio: Optional[float] = None
    min_seeding_hours: Optional[float] = None
    min_uploaded_gb: Optional[float] = None
    max_size_gb: Optional[float] = None
    include_categories: Optional[str] = None
    exclude_categories: Optional[str] = None
    include_tags: Optional[str] = None
    exclude_tags: Optional[str] = None
    delete_files: Optional[bool] = None

class TorrentAction(BaseModel):
    hashes: List[str]
    action: str  # pause, resume, delete, recheck, reannounce
    delete_files: bool = False

class AddTorrentRequest(BaseModel):
    urls: Optional[str] = None
    save_path: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[str] = None
    paused: bool = False

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


# ════════════════════════════════════════════════════════════════════════════════
# qBittorrent 管理器
# ════════════════════════════════════════════════════════════════════════════════

class QBManager:
    """管理多个qBittorrent实例"""
    
    def __init__(self):
        self.clients: Dict[int, qbittorrentapi.Client] = {}
        self.lock = threading.Lock()
    
    def connect(self, instance_id: int, host: str, username: str, password: str) -> Tuple[bool, str]:
        try:
            client = qbittorrentapi.Client(
                host=host,
                username=username,
                password=password,
                VERIFY_WEBUI_CERTIFICATE=False,
                REQUESTS_ARGS={'timeout': (5, 15)}
            )
            client.auth_log_in()
            with self.lock:
                self.clients[instance_id] = client
            return True, client.app.version
        except LoginFailed:
            return False, "登录失败，请检查用户名密码"
        except Exception as e:
            return False, str(e)
    
    def disconnect(self, instance_id: int):
        with self.lock:
            if instance_id in self.clients:
                try:
                    self.clients[instance_id].auth_log_out()
                except:
                    pass
                del self.clients[instance_id]
    
    def get_client(self, instance_id: int) -> Optional[qbittorrentapi.Client]:
        with self.lock:
            return self.clients.get(instance_id)
    
    def get_torrents(self, instance_id: int, status_filter: str = None) -> List[dict]:
        client = self.get_client(instance_id)
        if not client:
            return []
        try:
            torrents = client.torrents_info(status_filter=status_filter)
            return [self._torrent_to_dict(t) for t in torrents]
        except Exception as e:
            logger.error(f"获取种子列表失败: {e}")
            return []
    
    def _torrent_to_dict(self, t) -> dict:
        return {
            'hash': t.hash,
            'name': t.name,
            'state': t.state,
            'progress': t.progress,
            'size': t.total_size,
            'downloaded': t.completed or t.downloaded,
            'uploaded': t.uploaded,
            'ratio': t.ratio,
            'up_speed': t.upspeed,
            'dl_speed': t.dlspeed,
            'eta': t.eta,
            'added_on': t.added_on,
            'completed_on': t.completion_on,
            'seeding_time': getattr(t, 'seeding_time', 0),
            'time_active': getattr(t, 'time_active', 0),
            'tracker': t.tracker,
            'save_path': t.save_path,
            'category': t.category,
            'tags': t.tags,
            'up_limit': t.up_limit,
            'dl_limit': t.dl_limit,
            'num_seeds': t.num_seeds,
            'num_leechs': t.num_leechs,
            'availability': t.availability
        }
    
    def torrent_action(self, instance_id: int, hashes: List[str], action: str, delete_files: bool = False) -> bool:
        client = self.get_client(instance_id)
        if not client:
            return False
        try:
            if action == 'pause':
                client.torrents_pause(torrent_hashes=hashes)
            elif action == 'resume':
                client.torrents_resume(torrent_hashes=hashes)
            elif action == 'delete':
                client.torrents_delete(torrent_hashes=hashes, delete_files=delete_files)
            elif action == 'recheck':
                client.torrents_recheck(torrent_hashes=hashes)
            elif action == 'reannounce':
                client.torrents_reannounce(torrent_hashes=hashes)
            return True
        except Exception as e:
            logger.error(f"种子操作失败: {e}")
            return False
    
    def add_torrent(self, instance_id: int, urls: str = None, torrent_files: bytes = None,
                    save_path: str = None, category: str = None, tags: str = None,
                    paused: bool = False) -> bool:
        client = self.get_client(instance_id)
        if not client:
            return False
        try:
            if urls:
                client.torrents_add(
                    urls=urls,
                    save_path=save_path,
                    category=category,
                    tags=tags,
                    is_paused=paused
                )
            elif torrent_files:
                client.torrents_add(
                    torrent_files=torrent_files,
                    save_path=save_path,
                    category=category,
                    tags=tags,
                    is_paused=paused
                )
            return True
        except Exception as e:
            logger.error(f"添加种子失败: {e}")
            return False
    
    def set_upload_limit(self, instance_id: int, hashes: List[str], limit: int) -> bool:
        client = self.get_client(instance_id)
        if not client:
            return False
        try:
            client.torrents_set_upload_limit(limit, hashes)
            return True
        except Exception as e:
            logger.error(f"设置上传限速失败: {e}")
            return False
    
    def set_download_limit(self, instance_id: int, hashes: List[str], limit: int) -> bool:
        client = self.get_client(instance_id)
        if not client:
            return False
        try:
            client.torrents_set_download_limit(limit, hashes)
            return True
        except Exception as e:
            logger.error(f"设置下载限速失败: {e}")
            return False
    
    def get_torrent_properties(self, instance_id: int, torrent_hash: str) -> Optional[dict]:
        client = self.get_client(instance_id)
        if not client:
            return None
        try:
            props = client.torrents_properties(torrent_hash=torrent_hash)
            return dict(props)
        except:
            return None
    
    def get_global_info(self, instance_id: int) -> Optional[dict]:
        client = self.get_client(instance_id)
        if not client:
            return None
        try:
            info = client.transfer_info()
            return {
                'dl_speed': info.get('dl_info_speed', 0),
                'up_speed': info.get('up_info_speed', 0),
                'dl_total': info.get('dl_info_data', 0),
                'up_total': info.get('up_info_data', 0),
                'dht_nodes': info.get('dht_nodes', 0),
                'connection_status': info.get('connection_status', 'unknown')
            }
        except:
            return None


qb_manager = QBManager()


# ════════════════════════════════════════════════════════════════════════════════
# 动态限速引擎
# ════════════════════════════════════════════════════════════════════════════════

class TorrentLimitState:
    """单个种子的限速状态"""
    
    def __init__(self, torrent_hash: str):
        self.hash = torrent_hash
        self.name = ""
        self.site_id: Optional[int] = None
        
        # 周期信息
        self.cycle_start = 0.0
        self.cycle_start_uploaded = 0
        self.cycle_synced = False
        self.cycle_interval = 0.0
        self.cycle_index = 0
        self.jump_count = 0
        self.last_jump = 0.0
        
        # 时间信息
        self.time_added = 0.0
        self._publish_time: Optional[float] = None
        self._last_announce_time: Optional[float] = None
        
        # 上传信息
        self.initial_uploaded = 0
        self.total_size = 0
        self.total_uploaded_start = 0
        self.session_start_time = 0.0
        
        # 缓存
        self.cached_tl = 0.0
        self.cache_ts = 0.0
        self.prev_tl = 0.0
        
        # 限速状态
        self.last_up_limit = -1
        self.last_up_reason = ""
        self.last_dl_limit = -1
        
        # 控制器
        self.limit_controller = DynamicLimitController()
    
    def get_tl(self, now: float) -> float:
        if self._last_announce_time and self._last_announce_time > 0:
            interval = self.get_announce_interval()
            next_announce = self._last_announce_time + interval
            return max(0, next_announce - now)
        if self.cache_ts <= 0:
            return 9999
        return max(0, self.cached_tl - (now - self.cache_ts))
    
    def get_phase(self, now: float) -> str:
        return get_phase(self.get_tl(now), self.cycle_synced)
    
    def get_announce_interval(self) -> int:
        if self._publish_time and self._publish_time > 0:
            return estimate_announce_interval(self._publish_time)
        if self.time_added > 0:
            return estimate_announce_interval(self.time_added)
        return LimitConst.ANNOUNCE_INTERVAL_NEW
    
    def elapsed(self, now: float) -> float:
        return max(0, now - self.cycle_start) if self.cycle_start > 0 else 0
    
    def uploaded_in_cycle(self, current_uploaded: int) -> int:
        return max(0, current_uploaded - self.cycle_start_uploaded)
    
    def estimate_total(self, now: float, tl: float) -> float:
        e = self.elapsed(now)
        if 0 < tl < LimitConst.MAX_REANNOUNCE:
            return max(1, e + tl)
        if self.cycle_synced and self.cycle_interval > 0:
            return max(1, self.cycle_interval)
        return max(1, e)
    
    def get_real_avg_speed(self, current_uploaded: int) -> float:
        if self.session_start_time <= 0:
            return 0
        elapsed = time.time() - self.session_start_time
        if elapsed < 10:
            return 0
        uploaded = current_uploaded - self.total_uploaded_start
        return safe_div(uploaded, elapsed, 0)
    
    def new_cycle(self, now: float, uploaded: int, tl: float, is_jump: bool):
        if is_jump:
            self.jump_count += 1
            if self.jump_count >= 2 and self.last_jump > 0:
                self.cycle_interval = now - self.last_jump
                self.cycle_synced = True
            self.last_jump = now
            self.cycle_index += 1
            self.cycle_start_uploaded = uploaded
            self._last_announce_time = now
        elif self.time_added > 0 and (now - self.time_added) < self.get_announce_interval():
            self.cycle_start_uploaded = 0
        else:
            interval = self.get_announce_interval()
            elapsed_in_cycle = interval - tl if 0 < tl < interval else 0
            if elapsed_in_cycle > 60:
                avg_speed = self.limit_controller.kalman.speed
                if avg_speed > 0:
                    estimated_start = uploaded - int(avg_speed * elapsed_in_cycle)
                    self.cycle_start_uploaded = max(0, estimated_start)
                else:
                    self.cycle_start_uploaded = uploaded
            else:
                self.cycle_start_uploaded = uploaded
        
        self.cycle_start = now
        self.last_dl_limit = -1
        self.limit_controller.reset()


class LimitEngine:
    """动态限速引擎"""
    
    ACTIVE_STATES = frozenset({'downloading', 'seeding', 'uploading', 'forcedUP', 
                               'stalledUP', 'stalledDL', 'checkingUP', 'forcedDL', 
                               'checkingDL', 'metaDL'})
    
    def __init__(self):
        self.states: Dict[str, TorrentLimitState] = {}
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.lock = threading.Lock()
    
    def start(self):
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("🚀 限速引擎已启动")
    
    def stop(self):
        self.running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("🛑 限速引擎已停止")
    
    def _run_loop(self):
        """主循环"""
        while not self._stop_event.is_set():
            try:
                self._process_all_instances()
            except Exception as e:
                logger.error(f"限速引擎异常: {e}")
            
            self._stop_event.wait(1.0)
    
    def _process_all_instances(self):
        """处理所有qB实例"""
        conn = db.get_conn()
        c = conn.cursor()
        
        # 获取所有启用的实例
        c.execute('SELECT id, name FROM qb_instances WHERE enabled = 1')
        instances = c.fetchall()
        
        # 获取所有站点配置
        c.execute('SELECT id, tracker_keyword, target_speed_kib, safety_margin, enable_limit FROM sites')
        sites = {row[1]: {'id': row[0], 'target': row[2] * 1024 * row[3], 'enabled': row[4]} for row in c.fetchall()}
        
        conn.close()
        
        for instance_id, instance_name in instances:
            client = qb_manager.get_client(instance_id)
            if not client:
                continue
            
            try:
                self._process_instance(instance_id, client, sites)
            except Exception as e:
                logger.error(f"处理实例 {instance_name} 失败: {e}")
    
    def _process_instance(self, instance_id: int, client: qbittorrentapi.Client, sites: dict):
        """处理单个qB实例"""
        try:
            torrents = client.torrents_info(status_filter='active')
        except:
            return
        
        now = time.time()
        up_actions: Dict[int, List[str]] = {}
        
        for torrent in torrents:
            if torrent.state not in self.ACTIVE_STATES:
                continue
            
            # 匹配站点
            tracker = getattr(torrent, 'tracker', '') or ''
            site_config = None
            for keyword, config in sites.items():
                if keyword and keyword in tracker:
                    site_config = config
                    break
            
            if not site_config or not site_config['enabled']:
                continue
            
            # 获取或创建状态
            state_key = f"{instance_id}_{torrent.hash}"
            if state_key not in self.states:
                state = TorrentLimitState(torrent.hash)
                state.name = torrent.name
                state.site_id = site_config['id']
                state.time_added = getattr(torrent, 'added_on', 0) or 0
                state.initial_uploaded = getattr(torrent, 'uploaded', 0) or 0
                state.total_size = getattr(torrent, 'total_size', 0) or 0
                state.total_uploaded_start = state.initial_uploaded
                state.session_start_time = now
                self.states[state_key] = state
            
            state = self.states[state_key]
            
            # 计算限速
            limit = self._calculate_limit(state, torrent, site_config['target'], now, client)
            
            # 收集限速动作
            if limit != state.last_up_limit:
                up_actions.setdefault(limit, []).append(torrent.hash)
                state.last_up_limit = limit
        
        # 批量应用限速
        for limit, hashes in up_actions.items():
            try:
                client.torrents_set_upload_limit(limit, hashes)
            except:
                pass
    
    def _calculate_limit(self, state: TorrentLimitState, torrent, target: float, now: float, 
                         client: qbittorrentapi.Client) -> int:
        """计算上传限速"""
        current_speed = getattr(torrent, 'upspeed', 0) or 0
        total_uploaded = getattr(torrent, 'uploaded', 0) or 0
        
        # 记录速度
        state.limit_controller.record_speed(now, current_speed)
        
        # 获取属性
        try:
            props = client.torrents_properties(torrent_hash=torrent.hash)
            ra = props.get('reannounce', 0) or 0
            if 0 < ra < LimitConst.MAX_REANNOUNCE:
                state.cached_tl = ra
                state.cache_ts = now
        except:
            pass
        
        tl = state.get_tl(now)
        
        # 检测周期跳变
        is_jump = state.cycle_start > 0 and tl > state.prev_tl + 30
        
        # 新周期
        if state.cycle_start == 0 or is_jump:
            state.new_cycle(now, total_uploaded, tl, is_jump)
        
        state.prev_tl = tl
        
        # 计算限速
        elapsed = state.elapsed(now)
        uploaded = state.uploaded_in_cycle(total_uploaded)
        phase = state.get_phase(now)
        
        limit, reason, _ = state.limit_controller.calculate(
            target=target, uploaded=uploaded, time_left=tl,
            elapsed=elapsed, phase=phase, now=now
        )
        
        state.last_up_reason = reason
        
        # 超速保护
        real_speed = state.get_real_avg_speed(total_uploaded)
        if real_speed > target * 1.1:
            limit = LimitConst.MIN_LIMIT
        
        return limit
    
    def get_state(self, instance_id: int, torrent_hash: str) -> Optional[dict]:
        """获取种子限速状态"""
        state_key = f"{instance_id}_{torrent_hash}"
        state = self.states.get(state_key)
        if not state:
            return None
        
        now = time.time()
        return {
            'hash': state.hash,
            'phase': state.get_phase(now),
            'time_left': state.get_tl(now),
            'cycle_index': state.cycle_index,
            'cycle_synced': state.cycle_synced,
            'current_limit': state.last_up_limit,
            'limit_reason': state.last_up_reason,
            'elapsed': state.elapsed(now),
            'kalman_speed': state.limit_controller.kalman.speed
        }
    
    def get_all_states(self) -> List[dict]:
        """获取所有种子限速状态"""
        now = time.time()
        result = []
        for key, state in self.states.items():
            parts = key.split('_', 1)
            result.append({
                'instance_id': int(parts[0]),
                'hash': state.hash,
                'name': state.name,
                'phase': state.get_phase(now),
                'time_left': state.get_tl(now),
                'cycle_index': state.cycle_index,
                'cycle_synced': state.cycle_synced,
                'current_limit': state.last_up_limit,
                'limit_reason': state.last_up_reason
            })
        return result


limit_engine = LimitEngine()


# ════════════════════════════════════════════════════════════════════════════════
# RSS 管理器
# ════════════════════════════════════════════════════════════════════════════════

class RSSManager:
    """RSS订阅管理"""
    
    def __init__(self):
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start(self):
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("📡 RSS管理器已启动")
    
    def stop(self):
        self.running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
    
    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._check_tasks()
                self._check_feeds()
            except Exception as e:
                logger.error(f"RSS检查异常: {e}")
            
            self._stop_event.wait(300)  # 每5分钟检查一次
    
    def _check_feeds(self):
        conn = db.get_conn()
        c = conn.cursor()
        c.execute('SELECT id, name, url, qb_instance_id, save_path, category, auto_download, filter_include, filter_exclude FROM rss_feeds WHERE auto_download = 1')
        feeds = c.fetchall()
        conn.close()
        
        for feed in feeds:
            feed_id, name, url, qb_id, save_path, category, auto_dl, filter_inc, filter_exc = feed
            try:
                self._process_feed(feed_id, url, qb_id, save_path, category, filter_inc, filter_exc)
            except Exception as e:
                logger.error(f"处理RSS {name} 失败: {e}")

    def _check_tasks(self):
        conn = db.get_conn()
        c = conn.cursor()
        c.execute('''SELECT id, name, url, qb_instance_id, save_path, category, tags,
                    paused, interval_minutes, enabled, last_run
                    FROM rss_tasks WHERE enabled = 1''')
        tasks = c.fetchall()
        conn.close()

        for task in tasks:
            (task_id, name, url, qb_id, save_path, category, tags,
             paused, interval_minutes, enabled, last_run) = task
            if last_run and interval_minutes:
                if time.time() - last_run < interval_minutes * 60:
                    continue
            try:
                self._process_task(task_id, url, qb_id, save_path, category, tags, paused)
            except Exception as e:
                logger.error(f"处理RSS任务 {name} 失败: {e}")

    def _process_task(self, task_id: int, url: str, qb_id: int, save_path: str,
                      category: str, tags: str, paused: int):
        try:
            feed = feedparser.parse(url)
        except:
            return

        conn = db.get_conn()
        c = conn.cursor()
        c.execute('''SELECT id, name, include_keywords, exclude_keywords, save_path,
                    category, tags, paused FROM rss_task_rules WHERE task_id = ?''',
                  (task_id,))
        rules = c.fetchall()

        for entry in feed.entries[:50]:
            title = entry.get('title', '')
            link = entry.get('link', '')
            if not link:
                continue

            entry_hash = hashlib.md5(link.encode()).hexdigest()
            c.execute('SELECT id FROM rss_task_history WHERE task_id = ? AND entry_hash = ?',
                      (task_id, entry_hash))
            if c.fetchone():
                continue

            matched_rule = None
            if rules:
                for rule in rules:
                    rule_id, rule_name, inc, exc, rule_save, rule_cat, rule_tags, rule_paused = rule
                    if self._match_keywords(title, inc, exc):
                        matched_rule = {
                            'id': rule_id,
                            'save_path': rule_save,
                            'category': rule_cat,
                            'tags': rule_tags,
                            'paused': rule_paused
                        }
                        break
                if not matched_rule:
                    continue

            final_save = matched_rule['save_path'] if matched_rule and matched_rule['save_path'] else save_path
            final_cat = matched_rule['category'] if matched_rule and matched_rule['category'] else category
            final_tags = matched_rule['tags'] if matched_rule and matched_rule['tags'] else tags
            final_paused = bool(matched_rule['paused']) if matched_rule else bool(paused)

            if qb_manager.add_torrent(
                qb_id,
                urls=link,
                save_path=final_save,
                category=final_cat,
                tags=final_tags,
                paused=final_paused
            ):
                c.execute('''INSERT INTO rss_task_history
                          (task_id, rule_id, entry_hash, name, link, downloaded_at)
                          VALUES (?, ?, ?, ?, ?, ?)''',
                          (task_id, matched_rule['id'] if matched_rule else None,
                           entry_hash, title, link, time.time()))
                logger.info(f"📥 RSS任务添加种子: {title[:50]}")

        c.execute('UPDATE rss_tasks SET last_run = ? WHERE id = ?', (time.time(), task_id))
        conn.commit()
        conn.close()

    def _match_keywords(self, title: str, include: str, exclude: str) -> bool:
        if include:
            patterns = [p.strip() for p in include.split(',') if p.strip()]
            if patterns and not any(re.search(p, title, re.I) for p in patterns):
                return False
        if exclude:
            patterns = [p.strip() for p in exclude.split(',') if p.strip()]
            if patterns and any(re.search(p, title, re.I) for p in patterns):
                return False
        return True
    
    def _process_feed(self, feed_id: int, url: str, qb_id: int, save_path: str, 
                      category: str, filter_include: str, filter_exclude: str):
        try:
            feed = feedparser.parse(url)
        except:
            return
        
        conn = db.get_conn()
        c = conn.cursor()
        
        for entry in feed.entries[:20]:  # 只处理最新20个
            title = entry.get('title', '')
            link = entry.get('link', '')
            
            if not link:
                continue
            
            # 检查是否已下载
            entry_hash = hashlib.md5(link.encode()).hexdigest()
            c.execute('SELECT id FROM download_history WHERE hash = ?', (entry_hash,))
            if c.fetchone():
                continue
            
            # 应用过滤器
            if filter_include:
                patterns = [p.strip() for p in filter_include.split(',') if p.strip()]
                if patterns and not any(re.search(p, title, re.I) for p in patterns):
                    continue
            
            if filter_exclude:
                patterns = [p.strip() for p in filter_exclude.split(',') if p.strip()]
                if patterns and any(re.search(p, title, re.I) for p in patterns):
                    continue
            
            # 添加种子
            if qb_manager.add_torrent(qb_id, urls=link, save_path=save_path, category=category):
                c.execute('INSERT INTO download_history (hash, name, rss_feed_id, qb_instance_id, downloaded_at) VALUES (?, ?, ?, ?, ?)',
                         (entry_hash, title, feed_id, qb_id, time.time()))
                logger.info(f"📥 RSS添加种子: {title[:50]}")
        
        # 更新检查时间
        c.execute('UPDATE rss_feeds SET last_check = ? WHERE id = ?', (time.time(), feed_id))
        conn.commit()
        conn.close()
    
    def check_feed_now(self, feed_id: int) -> List[dict]:
        """立即检查RSS并返回条目"""
        conn = db.get_conn()
        c = conn.cursor()
        c.execute('SELECT url FROM rss_feeds WHERE id = ?', (feed_id,))
        row = c.fetchone()
        conn.close()
        
        if not row:
            return []
        
        try:
            feed = feedparser.parse(row[0])
            return [{
                'title': e.get('title', ''),
                'link': e.get('link', ''),
                'published': e.get('published', ''),
                'description': e.get('description', '')[:200] if e.get('description') else ''
            } for e in feed.entries[:50]]
        except:
            return []


rss_manager = RSSManager()


# ════════════════════════════════════════════════════════════════════════════════
# 删种管理器
# ════════════════════════════════════════════════════════════════════════════════

class DeleteManager:
    """删种任务管理"""

    def __init__(self):
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("🧹 删种管理器已启动")

    def stop(self):
        self.running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._check_tasks()
            except Exception as e:
                logger.error(f"删种检查异常: {e}")
            self._stop_event.wait(300)

    def _check_tasks(self):
        conn = db.get_conn()
        c = conn.cursor()
        c.execute('''SELECT id, name, qb_instance_id, interval_minutes, enabled, last_run
                    FROM delete_tasks WHERE enabled = 1''')
        tasks = c.fetchall()
        conn.close()

        for task in tasks:
            task_id, name, qb_id, interval_minutes, enabled, last_run = task
            if last_run and interval_minutes:
                if time.time() - last_run < interval_minutes * 60:
                    continue
            try:
                self._process_task(task_id, qb_id)
            except Exception as e:
                logger.error(f"处理删种任务 {name} 失败: {e}")

    def _process_task(self, task_id: int, qb_id: int):
        conn = db.get_conn()
        c = conn.cursor()
        c.execute('''SELECT id, name, min_ratio, min_seeding_hours, min_uploaded_gb, max_size_gb,
                    include_categories, exclude_categories, include_tags, exclude_tags, delete_files
                    FROM delete_task_rules WHERE task_id = ?''', (task_id,))
        rules = c.fetchall()
        conn.close()

        torrents = qb_manager.get_torrents(qb_id)
        for rule in rules:
            (rule_id, name, min_ratio, min_hours, min_uploaded_gb, max_size_gb,
             include_categories, exclude_categories, include_tags, exclude_tags, delete_files) = rule
            for torrent in torrents:
                if not self._match_delete_rule(
                    torrent,
                    min_ratio,
                    min_hours,
                    min_uploaded_gb,
                    max_size_gb,
                    include_categories,
                    exclude_categories,
                    include_tags,
                    exclude_tags
                ):
                    continue
                qb_manager.torrent_action(
                    qb_id,
                    [torrent['hash']],
                    'delete',
                    delete_files=bool(delete_files)
                )
        conn = db.get_conn()
        c = conn.cursor()
        c.execute('UPDATE delete_tasks SET last_run = ? WHERE id = ?', (time.time(), task_id))
        conn.commit()
        conn.close()

    def _match_delete_rule(self, torrent: dict, min_ratio: float, min_hours: float,
                           min_uploaded_gb: float, max_size_gb: float,
                           include_categories: str, exclude_categories: str,
                           include_tags: str, exclude_tags: str) -> bool:
        if min_ratio is not None and torrent.get('ratio', 0) < min_ratio:
            return False

        if min_hours is not None:
            seeding_time = torrent.get('seeding_time', 0)
            if seeding_time < min_hours * 3600:
                return False

        if min_uploaded_gb is not None:
            if torrent.get('uploaded', 0) < min_uploaded_gb * 1024 * 1024 * 1024:
                return False

        if max_size_gb is not None:
            if torrent.get('size', 0) > max_size_gb * 1024 * 1024 * 1024:
                return False

        if include_categories:
            allowed = {c.strip() for c in include_categories.split(',') if c.strip()}
            if allowed and torrent.get('category') not in allowed:
                return False

        if exclude_categories:
            blocked = {c.strip() for c in exclude_categories.split(',') if c.strip()}
            if blocked and torrent.get('category') in blocked:
                return False

        torrent_tags = {t.strip() for t in str(torrent.get('tags') or '').split(',') if t.strip()}

        if include_tags:
            required = {t.strip() for t in include_tags.split(',') if t.strip()}
            if required and not torrent_tags.intersection(required):
                return False

        if exclude_tags:
            blocked_tags = {t.strip() for t in exclude_tags.split(',') if t.strip()}
            if blocked_tags and torrent_tags.intersection(blocked_tags):
                return False

        return True


delete_manager = DeleteManager()


# ════════════════════════════════════════════════════════════════════════════════
# WebSocket 管理
# ════════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


ws_manager = ConnectionManager()


# ════════════════════════════════════════════════════════════════════════════════
# FastAPI 应用
# ════════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    logger.info(f"🚀 PTFlow v{VERSION} 启动中...")
    
    # 连接所有qB实例
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('SELECT id, name, host, username, password FROM qb_instances WHERE enabled = 1')
    instances = c.fetchall()
    conn.close()
    
    for inst_id, name, host, user, pwd in instances:
        success, msg = qb_manager.connect(inst_id, host, user, pwd)
        if success:
            logger.info(f"✅ 已连接 {name}: {msg}")
        else:
            logger.warning(f"⚠️ 连接 {name} 失败: {msg}")
    
    # 启动引擎
    limit_engine.start()
    rss_manager.start()
    delete_manager.start()
    
    yield
    
    # 关闭时
    logger.info("🛑 PTFlow 正在关闭...")
    limit_engine.stop()
    rss_manager.stop()
    delete_manager.stop()


app = FastAPI(title="PTFlow", version=VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


# 认证相关
def create_token(username: str) -> str:
    payload = {
        'sub': username,
        'exp': datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if not credentials:
        raise HTTPException(401, "未认证")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get('sub')
    except:
        raise HTTPException(401, "认证失败")


# ═══════════════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    conn = db.get_conn()
    c = conn.cursor()
    password_hash = hashlib.sha256(req.password.encode()).hexdigest()
    c.execute('SELECT id, username FROM users WHERE username = ? AND password_hash = ?', 
              (req.username, password_hash))
    user = c.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(401, "用户名或密码错误")
    
    return {
        'token': create_token(req.username),
        'username': req.username
    }


@app.post("/api/auth/change-password")
async def change_password(req: ChangePasswordRequest, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    old_hash = hashlib.sha256(req.old_password.encode()).hexdigest()
    c.execute('SELECT id FROM users WHERE username = ? AND password_hash = ?', (username, old_hash))
    if not c.fetchone():
        conn.close()
        raise HTTPException(400, "原密码错误")
    
    new_hash = hashlib.sha256(req.new_password.encode()).hexdigest()
    c.execute('UPDATE users SET password_hash = ? WHERE username = ?', (new_hash, username))
    conn.commit()
    conn.close()
    return {'success': True}


@app.get("/api/auth/me")
async def get_me(username: str = Depends(verify_token)):
    return {'username': username}


# qBittorrent 实例管理
@app.get("/api/qb/instances")
async def get_instances(username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('SELECT id, name, host, username, enabled, created_at FROM qb_instances')
    rows = c.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        inst_id, name, host, user, enabled, created = row
        client = qb_manager.get_client(inst_id)
        result.append({
            'id': inst_id,
            'name': name,
            'host': host,
            'username': user,
            'enabled': bool(enabled),
            'connected': client is not None,
            'created_at': created
        })
    return result


@app.post("/api/qb/instances")
async def create_instance(req: QBInstanceCreate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('INSERT INTO qb_instances (name, host, username, password, created_at) VALUES (?, ?, ?, ?, ?)',
              (req.name, req.host, req.username, req.password, time.time()))
    inst_id = c.lastrowid
    conn.commit()
    conn.close()
    
    # 尝试连接
    success, msg = qb_manager.connect(inst_id, req.host, req.username, req.password)
    
    return {
        'id': inst_id,
        'connected': success,
        'message': msg
    }


@app.put("/api/qb/instances/{instance_id}")
async def update_instance(instance_id: int, req: QBInstanceUpdate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    
    updates = []
    values = []
    if req.name is not None:
        updates.append('name = ?')
        values.append(req.name)
    if req.host is not None:
        updates.append('host = ?')
        values.append(req.host)
    if req.username is not None:
        updates.append('username = ?')
        values.append(req.username)
    if req.password is not None:
        updates.append('password = ?')
        values.append(req.password)
    if req.enabled is not None:
        updates.append('enabled = ?')
        values.append(1 if req.enabled else 0)
    
    if updates:
        values.append(instance_id)
        c.execute(f'UPDATE qb_instances SET {", ".join(updates)} WHERE id = ?', values)
        conn.commit()
    
    # 重新连接
    c.execute('SELECT host, username, password FROM qb_instances WHERE id = ?', (instance_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        qb_manager.disconnect(instance_id)
        if req.enabled is not False:
            qb_manager.connect(instance_id, row[0], row[1], row[2])
    
    return {'success': True}


@app.delete("/api/qb/instances/{instance_id}")
async def delete_instance(instance_id: int, username: str = Depends(verify_token)):
    qb_manager.disconnect(instance_id)
    
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM qb_instances WHERE id = ?', (instance_id,))
    conn.commit()
    conn.close()
    
    return {'success': True}


@app.post("/api/qb/instances/{instance_id}/connect")
async def connect_instance(instance_id: int, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('SELECT host, username, password FROM qb_instances WHERE id = ?', (instance_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(404, "实例不存在")
    
    success, msg = qb_manager.connect(instance_id, row[0], row[1], row[2])
    return {'success': success, 'message': msg}


# 种子管理
@app.get("/api/qb/instances/{instance_id}/torrents")
async def get_torrents(instance_id: int, status: str = None, username: str = Depends(verify_token)):
    torrents = qb_manager.get_torrents(instance_id, status)
    
    # 添加限速状态
    for t in torrents:
        state = limit_engine.get_state(instance_id, t['hash'])
        if state:
            t['limit_state'] = state
    
    return torrents


@app.post("/api/qb/instances/{instance_id}/torrents/action")
async def torrent_action(instance_id: int, req: TorrentAction, username: str = Depends(verify_token)):
    success = qb_manager.torrent_action(instance_id, req.hashes, req.action, req.delete_files)
    return {'success': success}


@app.post("/api/qb/instances/{instance_id}/torrents/add")
async def add_torrent(instance_id: int, req: AddTorrentRequest, username: str = Depends(verify_token)):
    success = qb_manager.add_torrent(
        instance_id, 
        urls=req.urls, 
        save_path=req.save_path,
        category=req.category,
        tags=req.tags,
        paused=req.paused
    )
    return {'success': success}


@app.get("/api/qb/instances/{instance_id}/info")
async def get_instance_info(instance_id: int, username: str = Depends(verify_token)):
    info = qb_manager.get_global_info(instance_id)
    if not info:
        raise HTTPException(404, "实例未连接")
    return info


# 站点管理
@app.get("/api/sites")
async def get_sites(username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('SELECT id, name, tracker_keyword, target_speed_kib, safety_margin, enable_limit, enable_dl_limit, enable_reannounce, created_at FROM sites')
    rows = c.fetchall()
    conn.close()
    
    return [{
        'id': r[0],
        'name': r[1],
        'tracker_keyword': r[2],
        'target_speed_kib': r[3],
        'safety_margin': r[4],
        'enable_limit': bool(r[5]),
        'enable_dl_limit': bool(r[6]),
        'enable_reannounce': bool(r[7]),
        'created_at': r[8]
    } for r in rows]


@app.post("/api/sites")
async def create_site(req: SiteCreate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''INSERT INTO sites (name, tracker_keyword, target_speed_kib, safety_margin, 
                enable_limit, enable_dl_limit, enable_reannounce, cookie, created_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (req.name, req.tracker_keyword, req.target_speed_kib, req.safety_margin,
               1 if req.enable_limit else 0, 1 if req.enable_dl_limit else 0,
               1 if req.enable_reannounce else 0, req.cookie, time.time()))
    site_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return {'id': site_id}


@app.put("/api/sites/{site_id}")
async def update_site(site_id: int, req: SiteUpdate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    
    updates = []
    values = []
    for field, value in req.dict(exclude_unset=True).items():
        if value is not None:
            if field in ('enable_limit', 'enable_dl_limit', 'enable_reannounce'):
                updates.append(f'{field} = ?')
                values.append(1 if value else 0)
            else:
                updates.append(f'{field} = ?')
                values.append(value)
    
    if updates:
        values.append(site_id)
        c.execute(f'UPDATE sites SET {", ".join(updates)} WHERE id = ?', values)
        conn.commit()
    
    conn.close()
    return {'success': True}


@app.delete("/api/sites/{site_id}")
async def delete_site(site_id: int, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM sites WHERE id = ?', (site_id,))
    conn.commit()
    conn.close()
    return {'success': True}


# RSS任务管理
@app.get("/api/rss/tasks")
async def get_rss_tasks(username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''SELECT id, name, url, qb_instance_id, save_path, category, tags,
                paused, interval_minutes, enabled, last_run, created_at
                FROM rss_tasks''')
    rows = c.fetchall()
    conn.close()

    return [{
        'id': r[0],
        'name': r[1],
        'url': r[2],
        'qb_instance_id': r[3],
        'save_path': r[4],
        'category': r[5],
        'tags': r[6],
        'paused': bool(r[7]),
        'interval_minutes': r[8],
        'enabled': bool(r[9]),
        'last_run': r[10],
        'created_at': r[11]
    } for r in rows]


@app.post("/api/rss/tasks")
async def create_rss_task(req: RSSTaskCreate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''INSERT INTO rss_tasks (name, url, qb_instance_id, save_path, category, tags,
                paused, interval_minutes, enabled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (req.name, req.url, req.qb_instance_id, req.save_path, req.category,
               req.tags, 1 if req.paused else 0, req.interval_minutes,
               1 if req.enabled else 0, time.time()))
    task_id = c.lastrowid
    conn.commit()
    conn.close()
    return {'id': task_id}


@app.put("/api/rss/tasks/{task_id}")
async def update_rss_task(task_id: int, req: RSSTaskUpdate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    updates = []
    values = []
    for field, value in req.dict(exclude_unset=True).items():
        if value is not None:
            if field in ('paused', 'enabled'):
                updates.append(f'{field} = ?')
                values.append(1 if value else 0)
            else:
                updates.append(f'{field} = ?')
                values.append(value)
    if updates:
        values.append(task_id)
        c.execute(f'UPDATE rss_tasks SET {", ".join(updates)} WHERE id = ?', values)
        conn.commit()
    conn.close()
    return {'success': True}


@app.delete("/api/rss/tasks/{task_id}")
async def delete_rss_task(task_id: int, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM rss_task_rules WHERE task_id = ?', (task_id,))
    c.execute('DELETE FROM rss_tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    return {'success': True}


@app.get("/api/rss/tasks/{task_id}/rules")
async def get_rss_task_rules(task_id: int, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''SELECT id, name, include_keywords, exclude_keywords, save_path, category,
                tags, paused, created_at FROM rss_task_rules WHERE task_id = ?''', (task_id,))
    rows = c.fetchall()
    conn.close()
    return [{
        'id': r[0],
        'name': r[1],
        'include_keywords': r[2],
        'exclude_keywords': r[3],
        'save_path': r[4],
        'category': r[5],
        'tags': r[6],
        'paused': bool(r[7]),
        'created_at': r[8]
    } for r in rows]


@app.post("/api/rss/tasks/{task_id}/rules")
async def create_rss_task_rule(task_id: int, req: RSSTaskRuleCreate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''INSERT INTO rss_task_rules (task_id, name, include_keywords, exclude_keywords,
                save_path, category, tags, paused, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (task_id, req.name, req.include_keywords, req.exclude_keywords,
               req.save_path, req.category, req.tags, 1 if req.paused else 0, time.time()))
    rule_id = c.lastrowid
    conn.commit()
    conn.close()
    return {'id': rule_id}


@app.put("/api/rss/rules/{rule_id}")
async def update_rss_task_rule(rule_id: int, req: RSSTaskRuleUpdate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    updates = []
    values = []
    for field, value in req.dict(exclude_unset=True).items():
        if value is not None:
            if field == 'paused':
                updates.append(f'{field} = ?')
                values.append(1 if value else 0)
            else:
                updates.append(f'{field} = ?')
                values.append(value)
    if updates:
        values.append(rule_id)
        c.execute(f'UPDATE rss_task_rules SET {", ".join(updates)} WHERE id = ?', values)
        conn.commit()
    conn.close()
    return {'success': True}


@app.delete("/api/rss/rules/{rule_id}")
async def delete_rss_task_rule(rule_id: int, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM rss_task_rules WHERE id = ?', (rule_id,))
    conn.commit()
    conn.close()
    return {'success': True}


@app.get("/api/rss/tasks/{task_id}/preview")
async def preview_rss_task(task_id: int, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('SELECT url FROM rss_tasks WHERE id = ?', (task_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return []
    try:
        feed = feedparser.parse(row[0])
        return [{
            'title': e.get('title', ''),
            'link': e.get('link', ''),
            'published': e.get('published', ''),
            'description': e.get('description', '')[:200] if e.get('description') else ''
        } for e in feed.entries[:50]]
    except:
        return []


# RSS管理
@app.get("/api/rss")
async def get_rss_feeds(username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''SELECT r.id, r.name, r.url, r.site_id, r.qb_instance_id, r.save_path, r.category,
                r.auto_download, r.filter_include, r.filter_exclude, r.last_check, r.created_at,
                s.name as site_name, q.name as qb_name
                FROM rss_feeds r
                LEFT JOIN sites s ON r.site_id = s.id
                LEFT JOIN qb_instances q ON r.qb_instance_id = q.id''')
    rows = c.fetchall()
    conn.close()
    
    return [{
        'id': r[0],
        'name': r[1],
        'url': r[2],
        'site_id': r[3],
        'qb_instance_id': r[4],
        'save_path': r[5],
        'category': r[6],
        'auto_download': bool(r[7]),
        'filter_include': r[8],
        'filter_exclude': r[9],
        'last_check': r[10],
        'created_at': r[11],
        'site_name': r[12],
        'qb_name': r[13]
    } for r in rows]


@app.post("/api/rss")
async def create_rss_feed(req: RSSFeedCreate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''INSERT INTO rss_feeds (name, url, site_id, qb_instance_id, save_path, category,
                auto_download, filter_include, filter_exclude, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (req.name, req.url, req.site_id, req.qb_instance_id, req.save_path, req.category,
               1 if req.auto_download else 0, req.filter_include, req.filter_exclude, time.time()))
    feed_id = c.lastrowid
    conn.commit()
    conn.close()
    
    return {'id': feed_id}


@app.put("/api/rss/{feed_id}")
async def update_rss_feed(feed_id: int, req: RSSFeedUpdate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    
    updates = []
    values = []
    for field, value in req.dict(exclude_unset=True).items():
        if value is not None:
            if field == 'auto_download':
                updates.append(f'{field} = ?')
                values.append(1 if value else 0)
            else:
                updates.append(f'{field} = ?')
                values.append(value)
    
    if updates:
        values.append(feed_id)
        c.execute(f'UPDATE rss_feeds SET {", ".join(updates)} WHERE id = ?', values)
        conn.commit()
    
    conn.close()
    return {'success': True}


@app.delete("/api/rss/{feed_id}")
async def delete_rss_feed(feed_id: int, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM rss_feeds WHERE id = ?', (feed_id,))
    conn.commit()
    conn.close()
    return {'success': True}


@app.get("/api/rss/{feed_id}/preview")
async def preview_rss_feed(feed_id: int, username: str = Depends(verify_token)):
    entries = rss_manager.check_feed_now(feed_id)
    return entries


# 删种任务管理
@app.get("/api/delete/tasks")
async def get_delete_tasks(username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''SELECT id, name, qb_instance_id, interval_minutes, enabled, last_run, created_at
                FROM delete_tasks''')
    rows = c.fetchall()
    conn.close()
    return [{
        'id': r[0],
        'name': r[1],
        'qb_instance_id': r[2],
        'interval_minutes': r[3],
        'enabled': bool(r[4]),
        'last_run': r[5],
        'created_at': r[6]
    } for r in rows]


@app.post("/api/delete/tasks")
async def create_delete_task(req: DeleteTaskCreate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''INSERT INTO delete_tasks (name, qb_instance_id, interval_minutes, enabled, created_at)
                VALUES (?, ?, ?, ?, ?)''',
              (req.name, req.qb_instance_id, req.interval_minutes, 1 if req.enabled else 0, time.time()))
    task_id = c.lastrowid
    conn.commit()
    conn.close()
    return {'id': task_id}


@app.put("/api/delete/tasks/{task_id}")
async def update_delete_task(task_id: int, req: DeleteTaskUpdate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    updates = []
    values = []
    for field, value in req.dict(exclude_unset=True).items():
        if value is not None:
            if field == 'enabled':
                updates.append(f'{field} = ?')
                values.append(1 if value else 0)
            else:
                updates.append(f'{field} = ?')
                values.append(value)
    if updates:
        values.append(task_id)
        c.execute(f'UPDATE delete_tasks SET {", ".join(updates)} WHERE id = ?', values)
        conn.commit()
    conn.close()
    return {'success': True}


@app.delete("/api/delete/tasks/{task_id}")
async def delete_delete_task(task_id: int, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM delete_task_rules WHERE task_id = ?', (task_id,))
    c.execute('DELETE FROM delete_tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    return {'success': True}


@app.get("/api/delete/tasks/{task_id}/rules")
async def get_delete_rules(task_id: int, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''SELECT id, name, min_ratio, min_seeding_hours, min_uploaded_gb, max_size_gb,
                include_categories, exclude_categories, include_tags, exclude_tags, delete_files, created_at
                FROM delete_task_rules WHERE task_id = ?''', (task_id,))
    rows = c.fetchall()
    conn.close()
    return [{
        'id': r[0],
        'name': r[1],
        'min_ratio': r[2],
        'min_seeding_hours': r[3],
        'min_uploaded_gb': r[4],
        'max_size_gb': r[5],
        'include_categories': r[6],
        'exclude_categories': r[7],
        'include_tags': r[8],
        'exclude_tags': r[9],
        'delete_files': bool(r[10]),
        'created_at': r[11]
    } for r in rows]


@app.post("/api/delete/tasks/{task_id}/rules")
async def create_delete_rule(task_id: int, req: DeleteTaskRuleCreate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('''INSERT INTO delete_task_rules (task_id, name, min_ratio, min_seeding_hours, min_uploaded_gb,
                max_size_gb, include_categories, exclude_categories, include_tags, exclude_tags,
                delete_files, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (task_id, req.name, req.min_ratio, req.min_seeding_hours, req.min_uploaded_gb,
               req.max_size_gb, req.include_categories, req.exclude_categories, req.include_tags,
               req.exclude_tags, 1 if req.delete_files else 0, time.time()))
    rule_id = c.lastrowid
    conn.commit()
    conn.close()
    return {'id': rule_id}


@app.put("/api/delete/rules/{rule_id}")
async def update_delete_rule(rule_id: int, req: DeleteTaskRuleUpdate, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    updates = []
    values = []
    for field, value in req.dict(exclude_unset=True).items():
        if value is not None:
            if field == 'delete_files':
                updates.append(f'{field} = ?')
                values.append(1 if value else 0)
            else:
                updates.append(f'{field} = ?')
                values.append(value)
    if updates:
        values.append(rule_id)
        c.execute(f'UPDATE delete_task_rules SET {", ".join(updates)} WHERE id = ?', values)
        conn.commit()
    conn.close()
    return {'success': True}


@app.delete("/api/delete/rules/{rule_id}")
async def delete_delete_rule(rule_id: int, username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    c.execute('DELETE FROM delete_task_rules WHERE id = ?', (rule_id,))
    conn.commit()
    conn.close()
    return {'success': True}


# 限速引擎状态
@app.get("/api/limit/states")
async def get_limit_states(username: str = Depends(verify_token)):
    return limit_engine.get_all_states()


@app.get("/api/limit/status")
async def get_limit_status(username: str = Depends(verify_token)):
    return {
        'running': limit_engine.running,
        'active_torrents': len(limit_engine.states)
    }


@app.post("/api/limit/start")
async def start_limit_engine(username: str = Depends(verify_token)):
    limit_engine.start()
    return {'success': True}


@app.post("/api/limit/stop")
async def stop_limit_engine(username: str = Depends(verify_token)):
    limit_engine.stop()
    return {'success': True}


# 统计信息
@app.get("/api/stats")
async def get_stats(username: str = Depends(verify_token)):
    conn = db.get_conn()
    c = conn.cursor()
    
    # qB实例统计
    c.execute('SELECT COUNT(*) FROM qb_instances')
    qb_count = c.fetchone()[0]
    
    # 站点统计
    c.execute('SELECT COUNT(*) FROM sites')
    site_count = c.fetchone()[0]
    
    # RSS统计
    c.execute('SELECT COUNT(*) FROM rss_feeds')
    rss_count = c.fetchone()[0]
    
    # 下载历史统计
    c.execute('SELECT COUNT(*) FROM download_history')
    download_count = c.fetchone()[0]
    
    conn.close()
    
    # 获取所有实例的汇总信息
    total_up_speed = 0
    total_dl_speed = 0
    for inst_id in qb_manager.clients.keys():
        info = qb_manager.get_global_info(inst_id)
        if info:
            total_up_speed += info.get('up_speed', 0)
            total_dl_speed += info.get('dl_speed', 0)
    
    return {
        'qb_instances': qb_count,
        'sites': site_count,
        'rss_feeds': rss_count,
        'downloads': download_count,
        'limit_engine_running': limit_engine.running,
        'active_limited_torrents': len(limit_engine.states),
        'total_up_speed': total_up_speed,
        'total_dl_speed': total_dl_speed
    }


# WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # 每秒推送状态更新
            await asyncio.sleep(1)
            
            stats = {
                'type': 'stats_update',
                'data': {
                    'limit_states': limit_engine.get_all_states()[:20],  # 限制数量
                    'timestamp': time.time()
                }
            }
            
            # 获取所有实例速度
            speeds = {}
            for inst_id, client in qb_manager.clients.items():
                info = qb_manager.get_global_info(inst_id)
                if info:
                    speeds[inst_id] = {
                        'up_speed': info.get('up_speed', 0),
                        'dl_speed': info.get('dl_speed', 0)
                    }
            stats['data']['instance_speeds'] = speeds
            
            await websocket.send_json(stats)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# 静态文件服务 (前端)
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
