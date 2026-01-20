#!/usr/bin/env python3
"""
qBit Smart Limit v11.0.0 PRO - 终极增强版

新增功能:
- [新增] Telegram 双向交互 (命令控制)
- [新增] SQLite 数据持久化 (重启不丢失状态)
- [新增] Cookie 失效检测和通知
- [新增] 动态修改配置 (无需重启)
- [优化] 美化所有通知消息
- [优化] 精简冗余代码
"""

import os
import sys
import time
import json
import signal
import logging
import threading
import queue
import re
import html
import sqlite3
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque, Tuple, List, Any
from datetime import datetime
from collections import deque
from functools import reduce

import requests
import qbittorrentapi
from qbittorrentapi.exceptions import APIConnectionError, LoginFailed

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════════
# 常量配置
# ════════════════════════════════════════════════════════════════════════════════
class C:
    VERSION = "11.0.0 PRO"
    
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
    
    MAX_REANNOUNCE = 86400
    PROPS_CACHE = {"finish": 0.2, "steady": 0.5, "catch": 1.0, "warmup": 2.0}
    LOG_INTERVAL = 20
    CONFIG_CHECK = 30
    
    ANNOUNCE_INTERVAL_NEW = 1800
    ANNOUNCE_INTERVAL_WEEK = 2700
    ANNOUNCE_INTERVAL_OLD = 3600
    
    SPEED_LIMIT = 50 * 1024 * 1024
    
    DL_LIMIT_MIN_TIME = 20
    DL_LIMIT_BUFFER = 30
    DL_LIMIT_MIN = 512
    DL_LIMIT_ADJUST_BUFFER = 60
    
    REANNOUNCE_WAIT_LIMIT = 5120
    REANNOUNCE_MIN_INTERVAL = 900
    REANNOUNCE_SPEED_SAMPLES = 300
    
    PEER_LIST_CHECK_INTERVAL = 300
    TID_SEARCH_INTERVAL = 60
    
    # 数据库相关
    DB_PATH = "qbit_smart_limit.db"
    DB_SAVE_INTERVAL = 180  # 每3分钟保存一次
    
    # TG Bot 轮询
    TG_POLL_INTERVAL = 2
    COOKIE_CHECK_INTERVAL = 3600  # 每小时检查一次cookie


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

def escape_html(t: str) -> str:
    return str(t).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def safe_div(a: float, b: float, default: float = 0) -> float:
    try:
        if b == 0 or abs(b) < 1e-10: return default
        return a / b
    except: return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))

def get_phase(tl: float, synced: bool) -> str:
    if not synced: return C.PHASE_WARMUP
    if tl <= C.FINISH_TIME: return C.PHASE_FINISH
    if tl <= C.STEADY_TIME: return C.PHASE_STEADY
    return C.PHASE_CATCH

def estimate_announce_interval(time_ref: float) -> int:
    age = time.time() - time_ref
    if age < 7 * 86400: return C.ANNOUNCE_INTERVAL_NEW
    elif age < 30 * 86400: return C.ANNOUNCE_INTERVAL_WEEK
    return C.ANNOUNCE_INTERVAL_OLD

def wall_time() -> float:
    return time.time()

def parse_speed_str(s: str) -> Optional[int]:
    """解析速度字符串，如 '100M' -> 102400 (KiB)"""
    s = s.strip().upper()
    match = re.match(r'^(\d+(?:\.\d+)?)\s*(K|M|G|KB|MB|GB|KIB|MIB|GIB)?$', s)
    if not match: return None
    num = float(match.group(1))
    unit = match.group(2) or 'K'
    multipliers = {'K': 1, 'KB': 1, 'KIB': 1, 'M': 1024, 'MB': 1024, 'MIB': 1024, 'G': 1048576, 'GB': 1048576, 'GIB': 1048576}
    return int(num * multipliers.get(unit, 1))

def setup_logging(level: str = "INFO") -> logging.Logger:
    log = logging.getLogger("qsl")
    log.setLevel(logging.DEBUG)
    for h in list(log.handlers):
        try: h.close()
        except: pass
    log.handlers.clear()
    
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    log.addHandler(console)
    
    try:
        os.makedirs("/var/log", exist_ok=True)
        fh = RotatingFileHandler("/var/log/qbit-smart-limit.log", maxBytes=10*1024*1024, backupCount=3)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        log.addHandler(fh)
    except: pass
    return log

# 日志环形缓冲区（用于 /log 命令）
class LogBuffer:
    def __init__(self, maxlen: int = 100):
        self._buffer: Deque[str] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
    
    def add(self, msg: str):
        with self._lock:
            self._buffer.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")
    
    def get_recent(self, n: int = 10) -> List[str]:
        with self._lock:
            return list(self._buffer)[-n:]

log_buffer = LogBuffer()
logger = setup_logging()

# 包装logger以同时写入buffer
class LoggerWrapper:
    def __init__(self, logger: logging.Logger, buffer: LogBuffer):
        self._logger = logger
        self._buffer = buffer
    
    def info(self, msg): self._logger.info(msg); self._buffer.add(f"[I] {msg}")
    def warning(self, msg): self._logger.warning(msg); self._buffer.add(f"[W] {msg}")
    def error(self, msg): self._logger.error(msg); self._buffer.add(f"[E] {msg}")
    def debug(self, msg): self._logger.debug(msg)

logger = LoggerWrapper(logger, log_buffer)

# ════════════════════════════════════════════════════════════════════════════════
# 核心控制器 (PID/Kalman/Tracker)
# ════════════════════════════════════════════════════════════════════════════════
class PIDController:
    def __init__(self):
        self.kp = 0.6; self.ki = 0.15; self.kd = 0.08
        self._integral = 0.0; self._last_error = 0.0; self._last_time = 0.0
        self._last_output = 1.0; self._initialized = False
        self._integral_limit = 0.3; self._derivative_filter = 0.0
    
    def set_phase(self, phase: str):
        params = C.PID_PARAMS.get(phase, C.PID_PARAMS['steady'])
        self.kp, self.ki, self.kd = params['kp'], params['ki'], params['kd']
    
    def update(self, setpoint: float, measured: float, now: float) -> float:
        error = safe_div(setpoint - measured, max(setpoint, 1), 0)
        if not self._initialized:
            self._last_error = error; self._last_time = now; self._initialized = True
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
        self._integral = 0.0; self._last_error = 0.0; self._last_time = 0.0
        self._last_output = 1.0; self._derivative_filter = 0.0; self._initialized = False


class ExtendedKalman:
    def __init__(self):
        self.speed = 0.0; self.accel = 0.0
        self.p00 = 1000.0; self.p01 = 0.0; self.p10 = 0.0; self.p11 = 1000.0
        self._last_time = 0.0; self._initialized = False
    
    def update(self, measurement: float, now: float) -> Tuple[float, float]:
        if not self._initialized:
            self.speed = measurement; self._last_time = now; self._initialized = True
            return measurement, 0.0
        dt = now - self._last_time
        if dt <= 0.01: return self.speed, self.accel
        self._last_time = now
        
        pred_speed = self.speed + self.accel * dt
        p00_pred = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11 + C.KALMAN_Q_SPEED
        p01_pred = self.p01 + dt * self.p11
        p10_pred = self.p10 + dt * self.p11
        p11_pred = self.p11 + C.KALMAN_Q_ACCEL
        
        s = p00_pred + C.KALMAN_R
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
        self.speed = 0.0; self.accel = 0.0; self.p00 = 1000.0
        self.p01 = 0.0; self.p10 = 0.0; self.p11 = 1000.0; self._initialized = False


class MultiWindowSpeedTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=1200)
    
    def record(self, now: float, speed: float):
        with self._lock: self._samples.append((now, speed))
    
    def get_weighted_avg(self, now: float, phase: str) -> float:
        weights = C.WINDOW_WEIGHTS.get(phase, C.WINDOW_WEIGHTS['steady'])
        with self._lock: samples = list(self._samples)
        
        total_weight = 0.0; weighted_sum = 0.0
        for window in C.SPEED_WINDOWS:
            win_samples = [s for t, s in samples if now - t <= window]
            if win_samples:
                avg = sum(win_samples) / len(win_samples)
                w = weights.get(window, 0.25)
                weighted_sum += avg * w; total_weight += w
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
        with self._lock: self._samples.clear()


class AdaptiveQuantizer:
    @staticmethod
    def quantize(limit: int, phase: str, current_speed: float, target: float, trend: float = 0) -> int:
        if limit <= 0: return limit
        base = C.QUANT_STEPS.get(phase, 1024)
        ratio = safe_div(current_speed, target, 1)
        
        if phase == 'finish': step = 256
        elif ratio > 1.2: step = base * 2
        elif ratio > 1.05: step = base
        elif ratio > 0.8: step = base // 2
        else: step = base
        
        if abs(trend) > 0.1: step = max(256, step // 2)
        step = int(clamp(step, 256, 8192))
        return max(C.MIN_LIMIT, int((limit + step // 2) // step) * step)


class PrecisionTracker:
    def __init__(self, window: int = 30):
        self._history: Deque[Tuple[float, str, float]] = deque(maxlen=window)
        self._phase_adj: Dict[str, float] = {'warmup': 1.0, 'catch': 1.0, 'steady': 1.0, 'finish': 1.0}
        self._global_adj = 1.0
        self._lock = threading.Lock()
    
    def record(self, ratio: float, phase: str, now: float):
        with self._lock:
            self._history.append((ratio, phase, now))
            self._update()
    
    def _update(self):
        if len(self._history) < 5: return
        phase_data: Dict[str, List[float]] = {}
        for ratio, phase, _ in self._history:
            phase_data.setdefault(phase, []).append(ratio)
        
        for phase, ratios in phase_data.items():
            if len(ratios) < 3: continue
            avg = sum(ratios) / len(ratios)
            if avg > 1.005: adj = 0.998
            elif avg > 1.001: adj = 0.999
            elif avg < 0.99: adj = 1.002
            elif avg < 0.995: adj = 1.001
            else: adj = 1.0
            self._phase_adj[phase] = clamp(self._phase_adj[phase] * adj, 0.92, 1.08)
        
        all_ratios = [r for r, _, _ in self._history]
        global_avg = sum(all_ratios) / len(all_ratios)
        if global_avg > 1.002: self._global_adj = clamp(self._global_adj * 0.999, 0.95, 1.05)
        elif global_avg < 0.995: self._global_adj = clamp(self._global_adj * 1.001, 0.95, 1.05)
    
    def get_adjustment(self, phase: str) -> float:
        with self._lock:
            return self._phase_adj.get(phase, 1.0) * self._global_adj

_precision_tracker = PrecisionTracker()


class PrecisionLimitController:
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
        if time_left <= 0: return -1, "汇报中", debug
        required_speed = need / time_left
        debug['required_speed'] = required_speed
        
        self.pid.set_phase(phase)
        pid_output = self.pid.update(target_total, uploaded, now)
        debug['pid_output'] = pid_output
        
        headroom = C.PID_PARAMS.get(phase, {}).get('headroom', 1.01)
        limit = -1; reason = ""
        
        if phase == 'finish':
            pred = debug['predicted_ratio']
            correction = max(0.8, 1 - (pred - 1) * 3) if pred > 1.002 else (min(1.2, 1 + (1 - pred) * 3) if pred < 0.998 else 1.0)
            limit = int(required_speed * pid_output * correction)
            reason = f"F:{required_speed/1024:.0f}K"
        elif phase == 'steady':
            if debug['predicted_ratio'] > 1.01: headroom = 1.0
            limit = int(required_speed * headroom * pid_output)
            reason = f"S:{required_speed/1024:.0f}K"
        elif phase == 'catch':
            if required_speed > adjusted_target * 5:
                limit = -1; reason = "C:欠速放开"
            else:
                limit = int(required_speed * headroom * pid_output)
                reason = f"C:{required_speed/1024:.0f}K"
        else:
            progress = safe_div(uploaded, target_total, 0)
            if progress >= 1.0:
                limit = C.MIN_LIMIT; reason = f"W:超{(progress-1)*100:.0f}%"
            elif progress >= 0.8:
                limit = int(required_speed * 1.01 * pid_output); reason = f"W:精控"
            elif progress >= 0.5:
                limit = int(required_speed * 1.05); reason = f"W:温控"
            else:
                limit = -1; reason = "W:预热"
        
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
            self._smooth_limit = int((1-factor) * self._smooth_limit + factor * new_limit)
        return self._smooth_limit
    
    def reset(self):
        self.kalman.reset()
        self.speed_tracker.clear()
        self.pid.reset()
        self._smooth_limit = -1


class SpeedTracker:
    def __init__(self, max_samples: int = 600):
        self._lock = threading.Lock()
        self._samples: Deque[Tuple[float, int, int, float, float]] = deque(maxlen=max_samples)
    
    def record(self, ts: float, u: int, d: int, us: float, ds: float):
        with self._lock: self._samples.append((ts, u, d, us, ds))
    
    def get_avg_speeds(self, window: float = 300) -> Tuple[float, float]:
        now = wall_time()
        with self._lock:
            samples = [(t, u, d, us, ds) for t, u, d, us, ds in self._samples if now - t <= window]
        if len(samples) < 2: return 0, 0
        first, last = samples[0], samples[-1]
        dt = last[0] - first[0]
        if dt <= 0: return 0, 0
        return safe_div(last[1]-first[1], dt, 0), safe_div(last[2]-first[2], dt, 0)
    
    def clear(self):
        with self._lock: self._samples.clear()

# ════════════════════════════════════════════════════════════════════════════════
# SQLite 数据持久化
# ════════════════════════════════════════════════════════════════════════════════
class Database:
    def __init__(self, db_path: str = C.DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # 种子状态表
            c.execute('''CREATE TABLE IF NOT EXISTS torrent_states (
                hash TEXT PRIMARY KEY,
                name TEXT,
                tid INTEGER,
                promotion TEXT,
                publish_time REAL,
                cycle_index INTEGER,
                cycle_start REAL,
                cycle_start_uploaded INTEGER,
                cycle_synced INTEGER,
                cycle_interval REAL,
                total_uploaded_start INTEGER,
                session_start_time REAL,
                last_announce_time REAL,
                updated_at REAL
            )''')
            
            # 统计表
            c.execute('''CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY,
                total_cycles INTEGER,
                success_cycles INTEGER,
                precision_cycles INTEGER,
                total_uploaded INTEGER,
                start_time REAL,
                updated_at REAL
            )''')
            
            # 配置运行时状态表
            c.execute('''CREATE TABLE IF NOT EXISTS runtime_config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at REAL
            )''')
            
            conn.commit()
            conn.close()
    
    def save_torrent_state(self, state: 'TorrentState'):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''INSERT OR REPLACE INTO torrent_states 
                (hash, name, tid, promotion, publish_time, cycle_index, cycle_start, 
                 cycle_start_uploaded, cycle_synced, cycle_interval, total_uploaded_start,
                 session_start_time, last_announce_time, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (state.hash, state.name, state.tid, state.promotion,
                 state.publish_time, state.cycle_index, state.cycle_start,
                 state.cycle_start_uploaded, 1 if state.cycle_synced else 0,
                 state.cycle_interval, state.total_uploaded_start,
                 state.session_start_time, state.last_announce_time, wall_time()))
            conn.commit()
            conn.close()
    
    def load_torrent_state(self, torrent_hash: str) -> Optional[dict]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT * FROM torrent_states WHERE hash = ?', (torrent_hash,))
            row = c.fetchone()
            conn.close()
            
            if not row: return None
            return {
                'hash': row[0], 'name': row[1], 'tid': row[2], 'promotion': row[3],
                'publish_time': row[4], 'cycle_index': row[5], 'cycle_start': row[6],
                'cycle_start_uploaded': row[7], 'cycle_synced': bool(row[8]),
                'cycle_interval': row[9], 'total_uploaded_start': row[10],
                'session_start_time': row[11], 'last_announce_time': row[12]
            }
    
    def save_stats(self, stats: 'Stats'):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''INSERT OR REPLACE INTO stats 
                (id, total_cycles, success_cycles, precision_cycles, total_uploaded, start_time, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?)''',
                (stats.total, stats.success, stats.precision, stats.uploaded, stats.start, wall_time()))
            conn.commit()
            conn.close()
    
    def load_stats(self) -> Optional[dict]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT * FROM stats WHERE id = 1')
            row = c.fetchone()
            conn.close()
            
            if not row: return None
            return {
                'total': row[1], 'success': row[2], 'precision': row[3],
                'uploaded': row[4], 'start': row[5]
            }
    
    def save_runtime_config(self, key: str, value: str):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('INSERT OR REPLACE INTO runtime_config (key, value, updated_at) VALUES (?, ?, ?)',
                      (key, value, wall_time()))
            conn.commit()
            conn.close()
    
    def get_runtime_config(self, key: str) -> Optional[str]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT value FROM runtime_config WHERE key = ?', (key,))
            row = c.fetchone()
            conn.close()
            return row[0] if row else None
    
    def delete_torrent_state(self, torrent_hash: str):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('DELETE FROM torrent_states WHERE hash = ?', (torrent_hash,))
            conn.commit()
            conn.close()
    
    def get_all_torrent_hashes(self) -> List[str]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT hash FROM torrent_states')
            rows = c.fetchall()
            conn.close()
            return [r[0] for r in rows]


# ════════════════════════════════════════════════════════════════════════════════
# U2 网页辅助 (支持优惠抓取和Cookie检测)
# ════════════════════════════════════════════════════════════════════════════════
class U2WebHelper:
    def __init__(self, cookie: str, proxy: str = ""):
        self.cookie = cookie
        self.proxy = proxy
        self.session = requests.Session()
        self.session.headers['User-Agent'] = f'qBit-Smart-Limit/{C.VERSION}'
        self.cookies = {'nexusphp_u2': cookie} if cookie else {}
        self.enabled = bool(cookie) and BS4_AVAILABLE
        self._lock = threading.Lock()
        self._last_cookie_check = 0
        self._cookie_valid = True
    
    def close(self):
        if self.session:
            try: self.session.close()
            except: pass
    
    def _request(self, url: str, timeout: int = 15) -> Optional[str]:
        try:
            proxies = {'http': self.proxy, 'https': self.proxy} if self.proxy else None
            resp = self.session.get(url, cookies=self.cookies, proxies=proxies, timeout=timeout)
            if resp.status_code == 200: return resp.text
        except Exception as e:
            logger.debug(f"请求失败 {url}: {e}")
        return None
    
    def check_cookie_valid(self) -> Tuple[bool, str]:
        """检查Cookie是否有效"""
        if not self.enabled:
            return False, "未配置Cookie"
        
        try:
            html = self._request('https://u2.dmhy.org/index.php', timeout=10)
            if not html:
                return False, "无法连接到U2"
            
            # 检查是否包含登录状态特征
            if 'logout.php' in html or '登出' in html or 'userdetails.php' in html:
                self._cookie_valid = True
                return True, "Cookie有效"
            else:
                self._cookie_valid = False
                return False, "Cookie已失效，请重新登录获取"
        except Exception as e:
            return False, f"检查失败: {e}"
    
    def is_cookie_valid(self) -> bool:
        return self._cookie_valid
    
    def search_tid_by_hash(self, torrent_hash: str) -> Optional[Tuple[int, Optional[float], str]]:
        if not self.enabled: return None
        try:
            url = f'https://u2.dmhy.org/torrents.php?search={torrent_hash}&search_area=5'
            html = self._request(url)
            if not html: return None
            
            with self._lock:
                soup = BeautifulSoup(html.replace('\n', ''), 'lxml')
                table = soup.select('table.torrents')
                if not table or len(table[0].contents) <= 1: return None
                
                row = table[0].contents[1]
                if not hasattr(row, 'contents') or len(row.contents) < 2: return None
                
                # 获取 TID
                try:
                    link = row.contents[1]
                    href = ""
                    if hasattr(link, 'find'):
                        a_tag = link.find('a')
                        href = a_tag.get('href', '') if a_tag else ''
                    
                    match = re.search(r'id=(\d+)', href)
                    if not match: return None
                    tid = int(match.group(1))
                except: return None

                # 获取发布时间
                publish_time = None
                try:
                    if len(row.contents) > 3:
                        time_cell = row.contents[3]
                        if hasattr(time_cell, 'find'):
                            time_elem = time_cell.find('time')
                            if time_elem:
                                date_str = time_elem.get('title') or time_elem.get_text(' ')
                                if date_str:
                                    dt = datetime.strptime(date_str.strip(), '%Y-%m-%d %H:%M:%S')
                                    publish_time = dt.timestamp()
                except: pass

                # 获取优惠信息
                promo_text = "无优惠"
                try:
                    promos = []
                    imgs = row.contents[1].find_all('img')
                    for img in imgs:
                        classes = img.get('class', [])
                        if not classes: continue
                        c_str = " ".join(classes) if isinstance(classes, list) else str(classes)
                        if 'pro_free2up' in c_str: promos.extend(['Free', '2x'])
                        elif 'pro_free' in c_str: promos.append('Free')
                        elif 'pro_2up' in c_str: promos.append('2x')
                        elif 'pro_50pct' in c_str: promos.append('50%')
                        elif 'pro_30pct' in c_str: promos.append('30%')
                        elif 'pro_custom' in c_str: promos.append('Custom')
                    
                    if promos:
                        promo_text = " + ".join(sorted(list(set(promos)), key=lambda x: len(x), reverse=True))
                except: pass

                logger.info(f"🔍 Hash {torrent_hash[:8]}... → tid {tid} | 优惠: {promo_text}")
                return (tid, publish_time, promo_text)

        except Exception as e:
            logger.debug(f"搜索 tid 失败: {e}")
            return None
    
    def get_peer_list_info(self, tid: int) -> Optional[dict]:
        if not self.enabled or not tid or tid < 0: return None
        try:
            url = f'https://u2.dmhy.org/viewpeerlist.php?id={tid}'
            html = self._request(url)
            if not html: return None
            
            with self._lock:
                soup = BeautifulSoup(html.replace('\n', ' '), 'lxml')
                tables = soup.find_all('table')
                result = {}
                
                for table in tables or []:
                    rows = table.find_all('tr')
                    for tr in rows:
                        if not tr.get('bgcolor'): continue
                        tds = tr.find_all('td')
                        if len(tds) < 2: continue
                        
                        try:
                            uploaded_str = tds[1].get_text(' ').strip()
                            if uploaded_str:
                                result['uploaded'] = self._parse_size(uploaded_str)
                        except: pass
                        
                        try:
                            if len(tds) > 10:
                                idle_str = tds[10].get_text(' ').strip()
                                if ':' in idle_str:
                                    parts = list(map(int, idle_str.split(':')))
                                    idle_seconds = reduce(lambda a, b: a * 60 + b, parts)
                                    result['last_announce'] = wall_time() - idle_seconds
                        except: pass
                        
                        if result: break
                    if result: break
                return result if result else None
        except: return None

    @staticmethod
    def _parse_size(size_str: str) -> int:
        try:
            parts = size_str.strip().split()
            if len(parts) != 2: return 0
            num = float(parts[0].replace(',', '.'))
            unit = parts[1]
            units = {'B': 0, 'KiB': 1, 'MiB': 2, 'GiB': 3, 'TiB': 4, 'PiB': 5}
            exp = units.get(unit, 0)
            return int(num * (1024 ** exp))
        except: return 0

# ════════════════════════════════════════════════════════════════════════════════
# Telegram 双向交互系统 (PRO版)
# ════════════════════════════════════════════════════════════════════════════════
class TelegramBot:
    """支持命令交互的 Telegram Bot"""
    
    def __init__(self, token: str, chat_id: str, controller: 'Controller' = None):
        self.enabled = bool(token and chat_id)
        self.token = token
        self.chat_id = str(chat_id).strip()
        self.controller = controller
        self.base_url = f"https://api.telegram.org/bot{token}" if token else ""
        
        self._queue: queue.Queue = queue.Queue(maxsize=100)
        self._last_update_id = 0
        self._last_send: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        
        # 运行时状态
        self.paused = False
        self.temp_target_kib: Optional[int] = None  # 临时目标速度
        
        if self.enabled:
            self._session = requests.Session()
            # 启动消息发送线程
            threading.Thread(target=self._send_worker, daemon=True, name="TG-Sender").start()
            # 启动命令监听线程
            threading.Thread(target=self._poll_worker, daemon=True, name="TG-Poller").start()
    
    def _html_sanitize(self, msg: str) -> str:
        """
        Sanitize message for Telegram HTML parse_mode.

        - Preserve Telegram-supported HTML tags (b/strong/i/em/u/ins/s/strike/del/code/pre/a/span/tg-spoiler/blockquote).
        - Escape unsupported tags like <速度> => &lt;速度&gt;
        - Escape stray '&' (not part of an entity) to avoid HTML parse errors.
        """
        if not msg:
            return msg

        # Escape stray '&' but keep existing entities like &lt; &amp; &#123;
        msg = re.sub(r'&(?![a-zA-Z]+;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', str(msg))

        if '<' not in msg:
            return msg

        allowed = {
            'b','strong','i','em','u','ins','s','strike','del',
            'code','pre','a','span','tg-spoiler','blockquote'
        }

        def repl(m: re.Match) -> str:
            full = m.group(0)
            inner = (m.group(1) or '').strip()
            if not inner:
                return html.escape(full)

            name = inner.lstrip('/').split()[0].lower()
            if name not in allowed:
                return html.escape(full)

            # Telegram: <a> needs href=
            if name == 'a' and not inner.startswith('/'):
                if re.search(r'\bhref\s*=', inner, flags=re.IGNORECASE):
                    return full
                return html.escape(full)

            # Telegram: <span> only for spoiler (class="tg-spoiler")
            if name == 'span' and not inner.startswith('/'):
                if re.search(r'tg-spoiler', inner, flags=re.IGNORECASE):
                    return full
                return html.escape(full)

            return full

        # Replace every <...> region with either allowed tag or escaped literal
        msg = re.sub(r'<([^<>]+)>', repl, msg)
        return msg

    def set_controller(self, controller: 'Controller'):
        self.controller = controller
    
    def close(self):
        self._stop.set()
    
    # ═══════════════════════════════════════════
    # 消息发送
    # ═══════════════════════════════════════════
    def _send_worker(self):
        while not self._stop.is_set():
            try:
                msg = self._queue.get(timeout=5)
                if not msg:
                    continue
                try:
                    resp = self._session.post(
                        f"{self.base_url}/sendMessage",
                        json={
                            "chat_id": self.chat_id,
                            "text": self._html_sanitize(msg),
                            "parse_mode": "HTML",
                            "disable_web_page_preview": True
                        },
                        timeout=20
                    )
                    if resp.status_code == 429:
                        retry = resp.json().get('parameters', {}).get('retry_after', 30)
                        logger.warning(f"⚠️ TG 限流! 暂停 {retry}s")
                        time.sleep(retry + 1)
                    elif resp.status_code != 200:
                        logger.warning(f"⚠️ TG发送失败 HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(3)  # 防封间隔
                except Exception as e:
                    logger.debug(f"TG发送失败: {e}")
                    time.sleep(5)
            except queue.Empty:
                pass
            except Exception:
                time.sleep(1)

    def send(self, msg: str, key: str = None, interval: int = 60):
        if not self.enabled: return
        if key:
            with self._lock:
                now = wall_time()
                if key in self._last_send and now - self._last_send[key] < max(10, interval):
                    return
                self._last_send[key] = now
        try: self._queue.put_nowait(msg)
        except: pass
    
    def send_immediate(self, msg: str):
        """立即发送（用于命令回复）"""
        if not self.enabled:
            return
        try:
            resp = self._session.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": self._html_sanitize(msg),
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                },
                timeout=15
            )
            if resp.status_code != 200:
                logger.warning(f"⚠️ TG回复失败 HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.debug(f"TG回复异常: {e}")

    def _poll_worker(self):
        """轮询获取用户命令"""
        # Optional: validate token once
        try:
            resp = self._session.get(f"{self.base_url}/getMe", timeout=10)
            if resp.status_code != 200:
                logger.warning(f"⚠️ TG getMe失败 HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.debug(f"TG getMe异常: {e}")

        while not self._stop.is_set():
            try:
                resp = self._session.get(
                    f"{self.base_url}/getUpdates",
                    params={
                        "offset": self._last_update_id + 1,
                        "timeout": 30,
                        "allowed_updates": ["message"]
                    },
                    timeout=35
                )
                if resp.status_code != 200:
                    logger.warning(f"⚠️ TG getUpdates失败 HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(5)
                    continue

                data = resp.json()
                for update in data.get('result', []):
                    self._last_update_id = update.get('update_id', self._last_update_id)
                    msg = update.get('message', {}) or {}
                    text = (msg.get('text') or '').strip()
                    chat_id = str((msg.get('chat') or {}).get('id', ''))

                    if not text:
                        continue

                    if text.startswith('/'):
                        logger.info(f"📩 TG命令: chat_id={chat_id} text={text}")

                    # 验证是否来自授权用户
                    if chat_id == self.chat_id and text.startswith('/'):
                        self._handle_command(text)
                    elif text.startswith('/'):
                        logger.warning(f"🚫 TG未授权chat_id: {chat_id} (期望 {self.chat_id})")
            except Exception as e:
                logger.debug(f"TG轮询异常: {e}")
                time.sleep(5)

            time.sleep(C.TG_POLL_INTERVAL)

    def _handle_command(self, text: str):
        """处理用户命令"""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        handlers = {
            '/start': self._cmd_start,
            '/help': self._cmd_help,
            '/status': self._cmd_status,
            '/pause': self._cmd_pause,
            '/resume': self._cmd_resume,
            '/limit': self._cmd_limit,
            '/log': self._cmd_log,
            '/cookie': self._cmd_cookie,
            '/config': self._cmd_config,
            '/stats': self._cmd_stats,
        }
        
        handler = handlers.get(cmd, self._cmd_unknown)
        try:
            handler(args)
        except Exception as e:
            self.send_immediate(f"❌ 命令执行出错: {e}")
    
    # ═══════════════════════════════════════════
    # 命令处理器
    # ═══════════════════════════════════════════
    def _cmd_start(self, args: str):
        self._cmd_help(args)
    
    def _cmd_help(self, args: str):
        msg = """🤖 <b>qBit Smart Limit 命令帮助</b>
━━━━━━━━━━━━━━━━━━━━━
📊 <b>状态查询</b>
├ /status - 查看所有种子状态
├ /stats - 查看统计信息
└ /log [n] - 查看最近n条日志

⚙️ <b>控制命令</b>
├ /pause - 暂停限速功能
├ /resume - 恢复限速功能
└ /limit <速度> - 设置目标速度
   例: /limit 100M 或 /limit 51200K

🔧 <b>配置管理</b>
├ /cookie - 检查U2 Cookie状态
└ /config <参数> <值> - 修改配置
   参数: qb_host, qb_user, qb_pass
━━━━━━━━━━━━━━━━━━━━━
💡 速度单位支持: K/M/G (KiB)"""
        self.send_immediate(msg)
    
    def _cmd_status(self, args: str):
        if not self.controller:
            self.send_immediate("❌ 控制器未初始化")
            return
        
        states = self.controller.states
        if not states:
            self.send_immediate("📭 当前没有正在监控的种子")
            return
        
        now = wall_time()
        lines = ["📊 <b>种子状态总览</b>", "━━━━━━━━━━━━━━━━━━━━━"]
        
        for h, state in list(states.items())[:10]:  # 最多显示10个
            name = escape_html(state.name[:25])
            phase = state.get_phase(now)
            tl = state.get_tl(now)
            
            # 获取当前速度
            speed = state.limit_controller.kalman.speed
            
            phase_emoji = {'warmup': '🔥', 'catch': '🏃', 'steady': '⚖️', 'finish': '🎯'}.get(phase, '❓')
            
            lines.append(f"{phase_emoji} <b>{name}</b>")
            lines.append(f"   ↑{fmt_speed(speed)} | ⏱{tl:.0f}s | 周期#{state.cycle_index}")
        
        if len(states) > 10:
            lines.append(f"\n... 还有 {len(states)-10} 个种子")
        
        # 添加状态信息
        lines.append("\n━━━━━━━━━━━━━━━━━━━━━")
        status = "⏸️ 已暂停" if self.paused else "▶️ 运行中"
        target = self.temp_target_kib or self.controller.config.target_speed_kib
        lines.append(f"状态: {status} | 目标: {fmt_speed(target * 1024)}")
        
        self.send_immediate("\n".join(lines))
    
    def _cmd_pause(self, args: str):
        self.paused = True
        self.send_immediate("""⏸️ <b>限速功能已暂停</b>
━━━━━━━━━━━━━━━━━━━━━
所有种子将以最大速度运行
发送 /resume 恢复限速""")
        logger.warning("⏸️ 用户暂停了限速功能")
    
    def _cmd_resume(self, args: str):
        self.paused = False
        self.send_immediate("""▶️ <b>限速功能已恢复</b>
━━━━━━━━━━━━━━━━━━━━━
种子将按目标速度限制""")
        logger.info("▶️ 用户恢复了限速功能")
    
    def _cmd_limit(self, args: str):
        if not args:
            current = self.temp_target_kib or (self.controller.config.target_speed_kib if self.controller else 0)
            self.send_immediate(f"🎯 当前目标速度: <code>{fmt_speed(current * 1024)}</code>\n用法: /limit <速度> (如 100M)")
            return
        
        new_limit = parse_speed_str(args)
        if not new_limit or new_limit <= 0:
            self.send_immediate("❌ 无效的速度值\n例: /limit 100M 或 /limit 51200K")
            return
        
        old_limit = self.temp_target_kib or (self.controller.config.target_speed_kib if self.controller else 0)
        self.temp_target_kib = new_limit
        
        self.send_immediate(f"""🎯 <b>目标速度已修改</b>
━━━━━━━━━━━━━━━━━━━━━
原速度: <code>{fmt_speed(old_limit * 1024)}</code>
新速度: <code>{fmt_speed(new_limit * 1024)}</code>
━━━━━━━━━━━━━━━━━━━━━
⚠️ 此为临时设置，重启后恢复
如需永久修改请编辑配置文件""")
        logger.info(f"🎯 用户修改目标速度: {fmt_speed(old_limit*1024)} → {fmt_speed(new_limit*1024)}")
    
    def _cmd_log(self, args: str):
        try:
            n = int(args) if args else 10
            n = min(max(1, n), 30)
        except:
            n = 10
        
        logs = log_buffer.get_recent(n)
        if not logs:
            self.send_immediate("📜 暂无日志记录")
            return
        
        msg = f"📜 <b>最近 {len(logs)} 条日志</b>\n━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "\n".join(f"<code>{escape_html(l)}</code>" for l in logs)
        self.send_immediate(msg)
    
    def _cmd_cookie(self, args: str):
        if not self.controller or not self.controller.u2_helper:
            self.send_immediate("❌ U2辅助功能未启用")
            return
        
        self.send_immediate("🔍 正在检查 Cookie 状态...")
        valid, msg = self.controller.u2_helper.check_cookie_valid()
        
        if valid:
            self.send_immediate(f"""✅ <b>Cookie 状态正常</b>
━━━━━━━━━━━━━━━━━━━━━
状态: {msg}
检查时间: {datetime.now().strftime('%H:%M:%S')}""")
        else:
            self.send_immediate(f"""❌ <b>Cookie 状态异常</b>
━━━━━━━━━━━━━━━━━━━━━
问题: {msg}
━━━━━━━━━━━━━━━━━━━━━
⚠️ 请尽快更新 Cookie！
登录 U2 后获取新的 nexusphp_u2 值""")
    
    def _cmd_config(self, args: str):
        if not args:
            self.send_immediate("""⚙️ <b>配置修改帮助</b>
━━━━━━━━━━━━━━━━━━━━━
用法: /config <参数> <值>

可用参数:
├ qb_host - qBittorrent 地址
├ qb_user - qBittorrent 用户名
└ qb_pass - qBittorrent 密码

示例: /config qb_host http://127.0.0.1:8080""")
            return
        
        parts = args.split(maxsplit=1)
        if len(parts) != 2:
            self.send_immediate("❌ 用法: /config <参数> <值>")
            return
        
        param, value = parts
        param = param.lower()
        
        valid_params = {'qb_host': 'host', 'qb_user': 'username', 'qb_pass': 'password'}
        if param not in valid_params:
            self.send_immediate(f"❌ 未知参数: {param}\n可用: qb_host, qb_user, qb_pass")
            return
        
        if self.controller and self.controller.db:
            self.controller.db.save_runtime_config(f"override_{valid_params[param]}", value)
            self.send_immediate(f"""✅ <b>配置已保存</b>
━━━━━━━━━━━━━━━━━━━━━
参数: {param}
新值: <code>{escape_html(value[:30])}</code>
━━━━━━━━━━━━━━━━━━━━━
⚠️ 需要重启脚本生效""")
        else:
            self.send_immediate("❌ 数据库未初始化")
    
    def _cmd_stats(self, args: str):
        if not self.controller:
            self.send_immediate("❌ 控制器未初始化")
            return
        
        stats = self.controller.stats
        runtime = wall_time() - stats.start
        
        success_rate = safe_div(stats.success, stats.total, 0) * 100
        precision_rate = safe_div(stats.precision, stats.total, 0) * 100
        
        self.send_immediate(f"""📈 <b>运行统计</b>
━━━━━━━━━━━━━━━━━━━━━
⏱️ 运行时长: <code>{fmt_duration(runtime)}</code>

📊 <b>周期统计</b>
├ 总周期数: <code>{stats.total}</code>
├ 达标率: <code>{success_rate:.1f}%</code> ({stats.success}/{stats.total})
└ 精准率: <code>{precision_rate:.1f}%</code> ({stats.precision}/{stats.total})

📤 <b>流量统计</b>
└ 总上传: <code>{fmt_size(stats.uploaded)}</code>""")
    
    def _cmd_unknown(self, args: str):
        self.send_immediate("❓ 未知命令，发送 /help 查看帮助")
    
    # ═══════════════════════════════════════════
    # 通知方法（美化版）
    # ═══════════════════════════════════════════

    def startup(self, config: 'Config', qb_version: str = "", u2_enabled: bool = False):
        if not self.enabled: return
        msg = f"""🚀 <b>qBit Smart Limit 已启动</b>
━━━━━━━━━━━━━━━━━━━━━
📌 <b>版本</b>: v{C.VERSION}

⚙️ <b>配置信息</b>
├ 🎯 目标速度: <code>{fmt_speed(config.target_bytes)}</code>
├ 🛡️ 安全边际: <code>{config.safety_margin:.0%}</code>
├ 🔄 汇报优化: {'✅' if config.enable_reannounce_opt else '❌'}
└ 📥 下载限速: {'✅' if config.enable_dl_limit else '❌'}

💻 <b>系统状态</b>
├ 🤖 qBittorrent: <code>{qb_version}</code>
├ 🌐 U2辅助: {'✅' if u2_enabled else '❌'}
└ 🕒 启动时间: <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>

━━━━━━━━━━━━━━━━━━━━━
💡 发送 /help 查看可用命令"""
        self.send(msg, "startup", 0)

    def monitor_start(self, info: dict):
        if not self.enabled: return
        h = info.get('hash', '')
        name = escape_html(info.get('name', 'Unknown'))
        total_size = info.get('total_size', 0)
        target = info.get('target', 0)
        promotion = info.get('promotion', '无优惠')
        tid = info.get('tid')
        
        if tid and tid > 0:
            linked_name = f'<a href="https://u2.dmhy.org/details.php?id={tid}&hit=1">{name}</a>'
        else:
            linked_name = f"<b>{name}</b>"
        
        msg = f"""🎬 <b>开始监控新任务</b>
━━━━━━━━━━━━━━━━━━━━━
📛 {linked_name}

📦 种子大小: <code>{fmt_size(total_size)}</code>
🎯 目标均速: <code>{fmt_speed(target)}</code>
🍪 优惠状态: <code>{promotion}</code>
📅 开始时间: <code>{datetime.now().strftime('%H:%M:%S')}</code>"""
        self.send(msg, f"start_{h}", 0)

    def check_finish(self, info: dict):
        """检查并通知下载完成"""
        if not self.enabled: return
        h = info.get('hash', '')
        progress = info.get('progress', 0)
        
        # 使用 controller 的 torrent_meta 来跟踪状态
        if not hasattr(self, '_finish_notified'):
            self._finish_notified = set()
        
        if progress >= 0.999 and h not in self._finish_notified:
            self._finish_notified.add(h)
            name = escape_html(info.get('name', 'Unknown'))
            total_up = info.get('total_uploaded', 0)
            total_dl = info.get('total_downloaded', 0)
            
            msg = f"""🎉 <b>种子下载完成!</b>
━━━━━━━━━━━━━━━━━━━━━
📛 <b>{name}</b>

⏱️ 完成时间: <code>{datetime.now().strftime('%H:%M:%S')}</code>

📊 <b>流量统计</b>
├ 📤 已上传: <code>{fmt_size(total_up)}</code>
└ 📥 已下载: <code>{fmt_size(total_dl)}</code>"""
            self.send(msg, f"finish_{h}", 0)

    def cycle_report(self, info: dict):
        if not self.enabled: return
        
        name = escape_html(info.get('name', 'Unknown')[:35])
        cycle_idx = info.get('idx', 0)
        uploaded = info.get('uploaded', 0)
        duration = info.get('duration', 0)
        ratio = info.get('ratio', 0)
        real_speed = info.get('real_speed', 0)
        progress_pct = info.get('progress_pct', 0)
        total_size = info.get('total_size', 0)
        total_up_life = info.get('total_uploaded_life', 0)
        total_dl_life = info.get('total_downloaded_life', 0)
        
        # 达标状态
        if ratio >= 0.99:
            status = "🎯 完美"
        elif ratio >= 0.95:
            status = "✅ 达标"
        elif ratio >= 0.90:
            status = "👍 良好"
        else:
            status = "⚠️ 欠速"
        
        left_size = total_size * (1 - progress_pct / 100)
        
        msg = f"""📊 <b>周期汇报 #{cycle_idx}</b>
━━━━━━━━━━━━━━━━━━━━━
📛 {name}

⚡ <b>本周期 ({fmt_duration(duration)})</b>
├ 📤 上传: <code>{fmt_size(uploaded)}</code>
├ 📈 均速: <code>{fmt_speed(real_speed)}</code>
└ 🎯 达标: {status} (<code>{ratio*100:.1f}%</code>)

📉 <b>整体进度</b>
├ ⏳ 进度: <code>{progress_pct:.1f}%</code>
├ 📦 剩余: <code>{fmt_size(left_size)}</code>
├ 📤 总上传: <code>{fmt_size(total_up_life)}</code>
└ 📥 总下载: <code>{fmt_size(total_dl_life)}</code>"""
        self.send(msg, f"cycle_{info.get('hash', '')}", 5)

    def overspeed_warning(self, name: str, real_speed: float, target: float, tid: int = None):
        msg = f"""🚨 <b>超速警告</b>
━━━━━━━━━━━━━━━━━━━━━
📛 {escape_html(name[:30])}

⚠️ 实际速度: <code>{fmt_speed(real_speed)}</code>
🎯 目标速度: <code>{fmt_speed(target)}</code>
📊 超速比例: <code>{real_speed/target*100:.0f}%</code>"""
        self.send(msg, f"overspeed_{name[:10]}", 120)

    def dl_limit_notify(self, name: str, dl_limit: float, reason: str, tid: int = None):
        msg = f"""📥 <b>下载限速启动</b>
━━━━━━━━━━━━━━━━━━━━━
📛 {escape_html(name[:30])}
🔒 限制速度: <code>{fmt_speed(dl_limit * 1024)}</code>
📝 原因: {reason}"""
        self.send(msg, f"dl_limit_{name[:10]}", 60)

    def reannounce_notify(self, name: str, reason: str, tid: int = None):
        msg = f"""🔄 <b>强制汇报</b>
━━━━━━━━━━━━━━━━━━━━━
📛 {escape_html(name[:30])}
📝 原因: {reason}"""
        self.send(msg, f"reannounce_{name[:10]}", 60)

    def cookie_invalid_notify(self):
        msg = """🔴 <b>Cookie 失效警告</b>
━━━━━━━━━━━━━━━━━━━━━
⚠️ U2 Cookie 已失效!

请尽快登录 U2 获取新的 Cookie
并更新配置文件中的 u2_cookie

━━━━━━━━━━━━━━━━━━━━━
🔧 更新后重启脚本生效"""
        self.send(msg, "cookie_invalid", 3600)

    def shutdown_report(self):
        if not self.enabled: return
        msg = f"""🛑 <b>脚本已停止</b>
━━━━━━━━━━━━━━━━━━━━━
⏱️ 停止时间: <code>{datetime.now().strftime('%H:%M:%S')}</code>"""
        self.send_immediate(msg)

# ════════════════════════════════════════════════════════════════════════════════
# 配置类
# ════════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    host: str
    username: str
    password: str
    target_speed_kib: int
    safety_margin: float = 0.98
    log_level: str = "INFO"
    target_tracker_keyword: str = ""
    exclude_tracker_keyword: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    max_physical_speed_kib: int = 0
    api_rate_limit: int = 20
    u2_cookie: str = ""
    proxy: str = ""
    peer_list_enabled: bool = True
    enable_dl_limit: bool = True
    enable_reannounce_opt: bool = True
    _mtime: float = 0
    
    @property
    def target_bytes(self) -> int:
        return max(1, int(self.target_speed_kib * 1024 * self.safety_margin))
    
    @property
    def max_physical_bytes(self) -> int:
        return (self.max_physical_speed_kib or 0) * 1024
    
    @classmethod
    def load(cls, path: str, db: 'Database' = None) -> Tuple[Optional['Config'], Optional[str]]:
        try:
            mtime = os.path.getmtime(path)
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            
            # 基础配置
            cfg = cls(
                host=str(d.get('host', '')).strip(),
                username=str(d.get('username', '')).strip(),
                password=str(d.get('password', '')).strip(),
                target_speed_kib=int(d.get('target_speed_kib', 0)),
                safety_margin=float(d.get('safety_margin', 0.98)),
                log_level=str(d.get('log_level', 'INFO')),
                target_tracker_keyword=str(d.get('target_tracker_keyword', '')).strip(),
                exclude_tracker_keyword=str(d.get('exclude_tracker_keyword', '')).strip(),
                telegram_bot_token=str(d.get('telegram_bot_token', '')).strip(),
                telegram_chat_id=str(d.get('telegram_chat_id', '')).strip(),
                max_physical_speed_kib=int(d.get('max_physical_speed_kib', 0) or 0),
                api_rate_limit=int(d.get('api_rate_limit', 20) or 20),
                u2_cookie=str(d.get('u2_cookie', '')).strip(),
                proxy=str(d.get('proxy', '')).strip(),
                peer_list_enabled=bool(d.get('peer_list_enabled', True)),
                enable_dl_limit=bool(d.get('enable_dl_limit', True)),
                enable_reannounce_opt=bool(d.get('enable_reannounce_opt', True)),
                _mtime=mtime
            )
            
            # 应用数据库中的运行时覆盖配置
            if db:
                for param, attr in [('host', 'host'), ('username', 'username'), ('password', 'password')]:
                    override = db.get_runtime_config(f"override_{attr}")
                    if override:
                        setattr(cfg, attr, override)
            
            return cfg, None
        except Exception as e:
            return None, str(e)


# ════════════════════════════════════════════════════════════════════════════════
# 种子状态类
# ════════════════════════════════════════════════════════════════════════════════
class TorrentState:
    def __init__(self, h: str):
        self.hash = h
        self.name = ""
        self._lock = threading.RLock()
        
        # 基础信息
        self._tid: Optional[int] = None
        self.tid_searched = False
        self.tid_search_time = 0.0
        self.tid_not_found = False
        self.promotion = "获取中..."
        self.monitor_notified = False
        
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
        self.dl_limited_this_cycle = False
        
        # 汇报状态
        self.last_reannounce = 0.0
        self.reannounced_this_cycle = False
        self.waiting_reannounce = False
        
        # 日志控制
        self.last_log = 0.0
        self.last_log_limit = -1
        self.last_props = 0.0
        self.report_sent = False
        
        # Peer list
        self.last_peer_list_check = 0.0
        self.peer_list_uploaded: Optional[int] = None
        
        # 控制器
        self.limit_controller = PrecisionLimitController()
        self.speed_tracker = SpeedTracker()
        self.last_debug: Dict[str, Any] = {}
    
    # 属性访问器
    @property
    def tid(self) -> Optional[int]:
        with self._lock: return self._tid
    
    @tid.setter
    def tid(self, value: Optional[int]):
        with self._lock: self._tid = value
    
    @property
    def publish_time(self) -> Optional[float]:
        with self._lock: return self._publish_time
    
    @publish_time.setter
    def publish_time(self, value: Optional[float]):
        with self._lock: self._publish_time = value
    
    @property
    def last_announce_time(self) -> Optional[float]:
        with self._lock: return self._last_announce_time
    
    @last_announce_time.setter
    def last_announce_time(self, value: Optional[float]):
        with self._lock: self._last_announce_time = value
    
    def get_tl(self, now: float) -> float:
        with self._lock:
            if self._last_announce_time and self._last_announce_time > 0:
                interval = self.get_announce_interval()
                next_announce = self._last_announce_time + interval
                return max(0, next_announce - now)
            if self.cache_ts <= 0: return 9999
            return max(0, self.cached_tl - (now - self.cache_ts))
    
    def get_phase(self, now: float) -> str:
        return get_phase(self.get_tl(now), self.cycle_synced)
    
    def get_announce_interval(self) -> int:
        with self._lock:
            if self._publish_time and self._publish_time > 0:
                return estimate_announce_interval(self._publish_time)
        if self.time_added > 0:
            return estimate_announce_interval(self.time_added)
        return C.ANNOUNCE_INTERVAL_NEW
    
    def elapsed(self, now: float) -> float:
        return max(0, now - self.cycle_start) if self.cycle_start > 0 else 0
    
    def this_time(self, now: float) -> float:
        return self.elapsed(now)
    
    def uploaded_in_cycle(self, current_uploaded: int) -> int:
        return max(0, current_uploaded - self.cycle_start_uploaded)
    
    def this_up(self, current_uploaded: int) -> int:
        return self.uploaded_in_cycle(current_uploaded)
    
    def estimate_total(self, now: float, tl: float) -> float:
        e = self.elapsed(now)
        if 0 < tl < C.MAX_REANNOUNCE: return max(1, e + tl)
        if self.cycle_synced and self.cycle_interval > 0: return max(1, self.cycle_interval)
        return max(1, e)
    
    def get_real_avg_speed(self, current_uploaded: int) -> float:
        if self.session_start_time <= 0: return 0
        elapsed = wall_time() - self.session_start_time
        if elapsed < 10: return 0
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
            with self._lock:
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
        self.report_sent = False
        self.dl_limited_this_cycle = False
        self.reannounced_this_cycle = False
        self.waiting_reannounce = False
        self.last_dl_limit = -1
        self.limit_controller.reset()
        self.speed_tracker.clear()
    
    def to_db_dict(self) -> dict:
        """转换为可存储的字典"""
        return {
            'hash': self.hash, 'name': self.name, 'tid': self.tid,
            'promotion': self.promotion, 'publish_time': self.publish_time,
            'cycle_index': self.cycle_index, 'cycle_start': self.cycle_start,
            'cycle_start_uploaded': self.cycle_start_uploaded,
            'cycle_synced': self.cycle_synced, 'cycle_interval': self.cycle_interval,
            'total_uploaded_start': self.total_uploaded_start,
            'session_start_time': self.session_start_time,
            'last_announce_time': self.last_announce_time
        }
    
    def load_from_db(self, data: dict):
        """从数据库加载状态"""
        if not data: return
        self.name = data.get('name', '')
        self.tid = data.get('tid')
        self.promotion = data.get('promotion', '获取中...')
        self.publish_time = data.get('publish_time')
        self.cycle_index = data.get('cycle_index', 0)
        self.cycle_start = data.get('cycle_start', 0)
        self.cycle_start_uploaded = data.get('cycle_start_uploaded', 0)
        self.cycle_synced = data.get('cycle_synced', False)
        self.cycle_interval = data.get('cycle_interval', 0)
        self.total_uploaded_start = data.get('total_uploaded_start', 0)
        self.session_start_time = data.get('session_start_time', 0)
        self.last_announce_time = data.get('last_announce_time')
        
        if self.tid:
            self.tid_searched = True


# ════════════════════════════════════════════════════════════════════════════════
# 辅助限速类
# ════════════════════════════════════════════════════════════════════════════════
class DownloadLimiter:
    @staticmethod
    def calc_dl_limit(state: TorrentState, total_uploaded: int, total_done: int, 
                      total_size: int, eta: int, up_speed: float, dl_speed: float, 
                      now: float) -> Tuple[int, str]:
        this_up = state.this_up(total_uploaded)
        this_time = state.this_time(now)
        if this_time < 2: return -1, ""
        
        avg_speed = this_up / this_time
        if avg_speed <= C.SPEED_LIMIT:
            if state.last_dl_limit > 0: return -1, "均值恢复"
            return -1, ""
        
        remaining = total_size - total_done
        if remaining <= 0: return -1, ""
        
        min_time = C.DL_LIMIT_MIN_TIME * (2 if state.last_up_limit > 0 else 1)
        
        if state.last_dl_limit <= 0:
            if 0 < eta <= min_time:
                denominator = this_up / C.SPEED_LIMIT - this_time + C.DL_LIMIT_BUFFER
                if denominator <= 0: return C.DL_LIMIT_MIN, "超速严重"
                dl_limit = remaining / denominator / 1024
                return max(C.DL_LIMIT_MIN, int(dl_limit)), f"均值超限"
        else:
            if avg_speed >= C.SPEED_LIMIT:
                if dl_speed / 1024 < 2 * state.last_dl_limit:
                    denominator = this_up / C.SPEED_LIMIT - this_time + C.DL_LIMIT_ADJUST_BUFFER
                    if denominator <= 0: return C.DL_LIMIT_MIN, "超速严重"
                    new_limit = remaining / denominator / 1024
                    new_limit = min(new_limit, 512000)
                    if new_limit > 1.5 * state.last_dl_limit:
                        new_limit = 1.5 * state.last_dl_limit
                    elif new_limit < state.last_dl_limit:
                        new_limit = new_limit / 1.5
                    return max(C.DL_LIMIT_MIN, int(new_limit)), "调整中"
                return state.last_dl_limit, "保持"
            else:
                return -1, "均值恢复"
        return -1, ""


class ReannounceOptimizer:
    @staticmethod
    def should_reannounce(state: TorrentState, total_uploaded: int, total_done: int,
                          total_size: int, up_speed: float, dl_speed: float, 
                          now: float) -> Tuple[bool, str]:
        if state.last_reannounce > 0 and now - state.last_reannounce < C.REANNOUNCE_MIN_INTERVAL:
            return False, ""
        
        this_up = state.this_up(total_uploaded)
        this_time = state.this_time(now)
        if this_time < 30: return False, ""
        
        avg_up, avg_dl = state.speed_tracker.get_avg_speeds(C.REANNOUNCE_SPEED_SAMPLES)
        if avg_up <= C.SPEED_LIMIT or avg_dl <= 0: return False, ""
        
        remaining = total_size - total_done
        if remaining <= 0: return False, ""
        
        announce_interval = state.get_announce_interval()
        complete_time = remaining / avg_dl + now
        perfect_time = complete_time - announce_interval * C.SPEED_LIMIT / avg_up
        
        if this_up / this_time > C.SPEED_LIMIT:
            earliest = (this_up - C.SPEED_LIMIT * this_time) / (45 * 1024 * 1024) + now
        else:
            earliest = now
        
        if earliest - (now - this_time) < C.REANNOUNCE_MIN_INTERVAL:
            return False, ""
        
        if earliest > perfect_time:
            if now >= earliest:
                if this_up / this_time > C.SPEED_LIMIT:
                    return True, "优化汇报"
            else:
                if earliest < perfect_time + 60:
                    state.waiting_reannounce = True
                    return False, "等待汇报"
        return False, ""
    
    @staticmethod
    def check_waiting_reannounce(state: TorrentState, total_uploaded: int, 
                                  now: float) -> Tuple[bool, str]:
        if not state.waiting_reannounce: return False, ""
        this_up = state.this_up(total_uploaded)
        this_time = state.this_time(now)
        if this_time < C.REANNOUNCE_MIN_INTERVAL: return False, ""
        avg_speed = this_up / this_time
        if avg_speed < C.SPEED_LIMIT:
            return True, "均值恢复"
        return False, ""


@dataclass
class Stats:
    start: float = field(default_factory=wall_time)
    total: int = 0
    success: int = 0
    precision: int = 0
    uploaded: int = 0
    
    def record(self, ratio: float, uploaded: int):
        self.total += 1
        self.uploaded += uploaded
        if ratio >= 0.95: self.success += 1
        if abs(ratio - 1) <= C.PRECISION_PERFECT: self.precision += 1
    
    def load_from_db(self, data: dict):
        if not data: return
        self.total = data.get('total', 0)
        self.success = data.get('success', 0)
        self.precision = data.get('precision', 0)
        self.uploaded = data.get('uploaded', 0)
        self.start = data.get('start', wall_time())

# ════════════════════════════════════════════════════════════════════════════════
# 主控制器
# ════════════════════════════════════════════════════════════════════════════════
class Controller:
    ACTIVE = frozenset({'downloading', 'seeding', 'uploading', 'forcedUP', 'stalledUP', 
                        'stalledDL', 'checkingUP', 'forcedDL', 'checkingDL', 'metaDL'})
    
    def __init__(self, path: str):
        global logger
        
        # 初始化数据库
        self.db = Database()
        
        # 加载配置
        cfg, err = Config.load(path, self.db)
        if err:
            print(f"❌ 配置错误: {err}")
            sys.exit(1)
        
        self.config = cfg
        self.config_path = path
        self.last_config_check = wall_time()
        
        # 重新设置日志
        logger = LoggerWrapper(setup_logging(cfg.log_level), log_buffer)
        
        # 初始化组件
        self.client: Optional[qbittorrentapi.Client] = None
        self.qb_version = ""
        self.states: Dict[str, TorrentState] = {}
        self.stats = Stats()
        
        # 从数据库恢复统计
        db_stats = self.db.load_stats()
        if db_stats:
            self.stats.load_from_db(db_stats)
            logger.info(f"📦 已从数据库恢复统计: {self.stats.total} 个周期")
        
        # 初始化 TG Bot
        self.notifier = TelegramBot(cfg.telegram_bot_token, cfg.telegram_chat_id, self)
        
        # 初始化 U2 辅助
        self.u2_helper: Optional[U2WebHelper] = None
        self.u2_enabled = False
        if cfg.peer_list_enabled and cfg.u2_cookie:
            if BS4_AVAILABLE:
                self.u2_helper = U2WebHelper(cfg.u2_cookie, cfg.proxy)
                self.u2_enabled = True
            else:
                logger.warning("⚠️ BeautifulSoup 未安装，U2功能已禁用")
        
        # 状态变量
        self.running = True
        self.modified_up: set = set()
        self.modified_dl: set = set()
        
        # API 限速
        self._api_times: Deque[float] = deque(maxlen=200)
        
        # TID 搜索队列
        self._pending_tid_searches: queue.Queue = queue.Queue()
        threading.Thread(target=self._tid_search_worker, daemon=True, name="TID-Search").start()
        
        # 数据库保存定时器
        self._last_db_save = wall_time()
        
        # Cookie 检查
        self._last_cookie_check = 0
        
        # 信号处理
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())
    
    def _tid_search_worker(self):
        """TID 搜索后台线程"""
        while self.running:
            try:
                item = self._pending_tid_searches.get(timeout=5)
                if item and self.u2_helper:
                    h, state = item
                    result = self.u2_helper.search_tid_by_hash(h)
                    if result:
                        tid, publish_time, promo = result
                        state.tid = tid
                        state.publish_time = publish_time
                        state.promotion = promo
                        state.tid_searched = True
                        # 立即保存到数据库
                        self.db.save_torrent_state(state)
                    else:
                        state.tid_not_found = True
                        state.tid_searched = True
                        state.promotion = "无优惠"
            except queue.Empty:
                pass
            except Exception as e:
                logger.debug(f"TID搜索异常: {e}")
    
    def _shutdown(self):
        """优雅关闭"""
        logger.info("🛑 正在停止服务...")
        self.running = False
        
        # 保存所有状态到数据库
        self._save_all_to_db()
        
        # 发送关闭通知
        self.notifier.shutdown_report()
        
        # 恢复限速设置
        if self.client:
            try:
                if self.modified_up:
                    self.client.torrents_set_upload_limit(-1, list(self.modified_up))
                if self.modified_dl:
                    self.client.torrents_set_download_limit(-1, list(self.modified_dl))
            except:
                pass
        
        # 关闭资源
        if self.u2_helper:
            self.u2_helper.close()
        self.notifier.close()
        
        sys.exit(0)
    
    def _save_all_to_db(self):
        """保存所有状态到数据库"""
        try:
            for state in self.states.values():
                self.db.save_torrent_state(state)
            self.db.save_stats(self.stats)
            logger.debug("💾 状态已保存到数据库")
        except Exception as e:
            logger.error(f"保存数据库失败: {e}")
    
    def _check_config(self, now: float):
        """检查配置更新"""
        if now - self.last_config_check < C.CONFIG_CHECK:
            return
        self.last_config_check = now
        
        try:
            mtime = os.path.getmtime(self.config_path)
            if mtime > self.config._mtime:
                new_cfg, err = Config.load(self.config_path, self.db)
                if not err:
                    self.config = new_cfg
                    logger.info("📝 配置已重新加载")
        except:
            pass
        
        # 定期保存数据库
        if now - self._last_db_save > C.DB_SAVE_INTERVAL:
            self._save_all_to_db()
            self._last_db_save = now
        
        # 定期检查 Cookie
        if self.u2_helper and now - self._last_cookie_check > C.COOKIE_CHECK_INTERVAL:
            self._last_cookie_check = now
            valid, msg = self.u2_helper.check_cookie_valid()
            if not valid:
                logger.warning(f"⚠️ Cookie 状态异常: {msg}")
                self.notifier.cookie_invalid_notify()
    
    def _connect(self):
        """连接 qBittorrent"""
        for i in range(5):
            try:
                self.client = qbittorrentapi.Client(
                    host=self.config.host,
                    username=self.config.username,
                    password=self.config.password,
                    VERIFY_WEBUI_CERTIFICATE=False,
                    REQUESTS_ARGS={'timeout': (5, 15)}
                )
                self.client.auth_log_in()
                self.qb_version = self.client.app.version
                logger.info(f"✅ 已连接 qBittorrent {self.qb_version}")
                return
            except LoginFailed:
                logger.error("❌ 登录失败，请检查用户名密码")
                sys.exit(1)
            except Exception as e:
                if i < 4:
                    logger.warning(f"连接失败，重试中... ({i+1}/5)")
                    time.sleep(2 ** i)
                else:
                    logger.error(f"❌ 无法连接: {e}")
                    raise
    
    def _api_ok(self, now: float) -> bool:
        """API 限速检查"""
        limit = self.config.api_rate_limit
        if limit <= 0:
            return True
        while self._api_times and now - self._api_times[0] > 1:
            self._api_times.popleft()
        if len(self._api_times) >= limit:
            return False
        self._api_times.append(now)
        return True
    
    def _get_props(self, h: str, state: TorrentState, now: float, force: bool = False) -> Optional[dict]:
        """获取种子属性"""
        phase = state.get_phase(now)
        cache = C.PROPS_CACHE.get(phase, 1.0)
        if not force and state.last_props > 0 and now - state.last_props < cache:
            return None
        if not force and not self._api_ok(now):
            return None
        try:
            props = self.client.torrents_properties(torrent_hash=h)
            state.last_props = now
            return props
        except Exception as e:
            logger.debug(f"获取属性失败: {e}")
            return None
    
    def _should_manage(self, torrent: Any) -> bool:
        """检查是否需要管理该种子"""
        tracker = getattr(torrent, 'tracker', '') or ''
        if self.config.exclude_tracker_keyword and self.config.exclude_tracker_keyword in tracker:
            return False
        if self.config.target_tracker_keyword and self.config.target_tracker_keyword not in tracker:
            return False
        return True
    
    def _maybe_search_tid(self, state: TorrentState, now: float):
        """尝试搜索 TID"""
        if not self.u2_helper or not self.u2_helper.enabled:
            return
        if state.tid or state.tid_searched:
            return
        if state.tid_not_found and now - state.tid_search_time < 3600:
            return
        if now - state.tid_search_time < C.TID_SEARCH_INTERVAL:
            return
        state.tid_search_time = now
        try:
            self._pending_tid_searches.put_nowait((state.hash, state))
        except:
            pass
    
    def _maybe_check_peer_list(self, state: TorrentState, now: float):
        """检查 peer list"""
        if not self.u2_helper or not self.u2_helper.enabled:
            return
        if not state.tid or state.tid < 0:
            return
        if now - state.last_peer_list_check < C.PEER_LIST_CHECK_INTERVAL:
            return
        state.last_peer_list_check = now
        
        def check():
            try:
                info = self.u2_helper.get_peer_list_info(state.tid)
                if info:
                    if 'last_announce' in info:
                        state.last_announce_time = info['last_announce']
                    if 'uploaded' in info:
                        state.peer_list_uploaded = info['uploaded']
            except Exception as e:
                logger.debug(f"peer list 检查失败: {e}")
        
        threading.Thread(target=check, daemon=True).start()
    
    def _do_reannounce(self, state: TorrentState, reason: str):
        """执行强制汇报"""
        try:
            self.client.torrents_reannounce(torrent_hashes=state.hash)
            state.last_reannounce = wall_time()
            state.reannounced_this_cycle = True
            state.waiting_reannounce = False
            state.last_announce_time = wall_time()
            logger.warning(f"[{state.name[:16]}] 🔄 强制汇报: {reason}")
            self.notifier.reannounce_notify(state.name, reason, state.tid)
        except Exception as e:
            logger.debug(f"强制汇报失败: {e}")
    
    def _get_effective_target(self) -> int:
        """获取有效的目标速度（考虑临时修改）"""
        if self.notifier.temp_target_kib:
            return int(self.notifier.temp_target_kib * 1024 * self.config.safety_margin)
        return self.config.target_bytes
    
    def _calc_upload_limit(self, state: TorrentState, torrent: Any, now: float, tl: float) -> Tuple[int, str]:
        """计算上传限速"""
        # 检查是否暂停
        if self.notifier.paused:
            return -1, "已暂停"
        
        target = self._get_effective_target()
        max_phy = self.config.max_physical_bytes
        current = getattr(torrent, 'upspeed', 0) or 0
        total_uploaded = getattr(torrent, 'uploaded', 0) or 0
        
        state.limit_controller.record_speed(now, current)
        real_speed = state.get_real_avg_speed(total_uploaded)
        
        # 超速保护
        if real_speed > C.SPEED_LIMIT * 1.05:
            logger.warning(f"[{state.name[:15]}] ⚠️ 超速 {fmt_speed(real_speed)}!")
            self.notifier.overspeed_warning(state.name, real_speed, target, state.tid)
            return C.MIN_LIMIT, "超速刹车"
        
        if state.waiting_reannounce:
            return C.REANNOUNCE_WAIT_LIMIT * 1024, "等待汇报"
        
        elapsed = state.elapsed(now)
        uploaded = state.uploaded_in_cycle(total_uploaded)
        phase = state.get_phase(now)
        precision_adj = _precision_tracker.get_adjustment(phase)
        
        limit, reason, debug = state.limit_controller.calculate(
            target=target, uploaded=uploaded, time_left=tl,
            elapsed=elapsed, phase=phase, now=now, precision_adj=precision_adj
        )
        state.last_debug = debug
        
        # 物理限速
        if max_phy > 0:
            if limit == -1:
                limit = int(max_phy)
            elif limit > max_phy:
                limit = int(max_phy)
        
        # 进度保护
        progress = safe_div(uploaded, target * state.estimate_total(now, tl), 0)
        if progress >= C.PROGRESS_PROTECT and current > target * C.SPEED_PROTECT_RATIO:
            protect = int(target * C.SPEED_PROTECT_LIMIT)
            if limit == -1 or limit > protect:
                limit = protect
                reason = f"保护"
        
        return limit, reason
    
    def _calc_download_limit(self, state: TorrentState, torrent: Any, now: float) -> Tuple[int, str]:
        """计算下载限速"""
        if not self.config.enable_dl_limit or self.notifier.paused:
            return -1, ""
        
        total_uploaded = getattr(torrent, 'uploaded', 0) or 0
        total_done = getattr(torrent, 'completed', 0) or getattr(torrent, 'downloaded', 0) or 0
        total_size = getattr(torrent, 'total_size', 0) or state.total_size or 0
        eta = getattr(torrent, 'eta', 0) or 0
        up_speed = getattr(torrent, 'upspeed', 0) or 0
        dl_speed = getattr(torrent, 'dlspeed', 0) or 0
        
        if total_size <= 0:
            return -1, ""
        
        torrent_state = getattr(torrent, 'state', '')
        if 'download' not in torrent_state.lower() and 'stalled' not in torrent_state.lower():
            if state.last_dl_limit > 0:
                return -1, "完成"
            return -1, ""
        
        return DownloadLimiter.calc_dl_limit(
            state, total_uploaded, total_done, total_size, eta, up_speed, dl_speed, now
        )
    
    def _check_reannounce(self, state: TorrentState, torrent: Any, now: float):
        """检查是否需要强制汇报"""
        if not self.config.enable_reannounce_opt or self.notifier.paused:
            return
        
        total_uploaded = getattr(torrent, 'uploaded', 0) or 0
        total_done = getattr(torrent, 'completed', 0) or getattr(torrent, 'downloaded', 0) or 0
        total_size = getattr(torrent, 'total_size', 0) or state.total_size or 0
        up_speed = getattr(torrent, 'upspeed', 0) or 0
        dl_speed = getattr(torrent, 'dlspeed', 0) or 0
        
        if total_size <= 0:
            return
        
        if state.waiting_reannounce:
            should, reason = ReannounceOptimizer.check_waiting_reannounce(state, total_uploaded, now)
            if should:
                self._do_reannounce(state, reason)
                return
        
        should, reason = ReannounceOptimizer.should_reannounce(
            state, total_uploaded, total_done, total_size, up_speed, dl_speed, now
        )
        if should:
            self._do_reannounce(state, reason)
    
    def _report(self, state: TorrentState, torrent: Any, now: float):
        """周期汇报"""
        if state.report_sent:
            return
        state.report_sent = True
        
        target = self._get_effective_target()
        duration = max(1, state.elapsed(now))
        total_uploaded = getattr(torrent, 'uploaded', 0) or 0
        uploaded = state.uploaded_in_cycle(total_uploaded)
        speed = safe_div(uploaded, duration, 0)
        ratio = safe_div(speed, target, 0)
        real_speed = state.get_real_avg_speed(total_uploaded)
        phase = state.get_phase(now)
        
        _precision_tracker.record(ratio, phase, now)
        self.stats.record(ratio, uploaded)
        
        total_size = getattr(torrent, 'total_size', 0) or state.total_size
        total_done = getattr(torrent, 'completed', 0) or getattr(torrent, 'downloaded', 0) or 0
        progress_pct = safe_div(total_done, total_size, 0) * 100 if total_size > 0 else 0
        
        # 日志输出
        dev = abs(ratio - 1)
        if dev <= C.PRECISION_PERFECT:
            g = "🎯"
        elif dev <= C.PRECISION_GOOD:
            g = "✅"
        elif ratio >= 0.95:
            g = "👍"
        else:
            g = "⚠️"
        
        extra = ""
        if state.dl_limited_this_cycle:
            extra += " 📥"
        if state.reannounced_this_cycle:
            extra += " 🔄"
        
        logger.info(f"[{torrent.name[:16]}] {g} 汇报 ↑{fmt_speed(speed)}({ratio*100:.1f}%){extra}")
        
        # TG 通知
        self.notifier.cycle_report({
            'name': torrent.name, 'hash': state.hash,
            'speed': speed, 'real_speed': real_speed, 'target': target,
            'ratio': ratio, 'uploaded': uploaded, 'duration': duration,
            'idx': state.cycle_index, 'tid': state.tid,
            'total_size': total_size, 'total_uploaded_life': total_uploaded,
            'total_downloaded_life': total_done, 'progress_pct': progress_pct
        })
    
    def _process(self, torrent: Any, now: float, up_actions: Dict[int, List[str]], 
                 dl_actions: Dict[int, List[str]]) -> float:
        """处理单个种子"""
        h = torrent.hash
        if not self._should_manage(torrent):
            return 9999
        
        total_uploaded = getattr(torrent, 'uploaded', 0) or 0
        total_downloaded = getattr(torrent, 'completed', 0) or getattr(torrent, 'downloaded', 0) or 0
        time_added = getattr(torrent, 'added_on', 0) or 0
        up_speed = getattr(torrent, 'upspeed', 0) or 0
        dl_speed = getattr(torrent, 'dlspeed', 0) or 0
        
        # 初始化或恢复状态
        if h not in self.states:
            state = TorrentState(h)
            
            # 尝试从数据库恢复
            db_data = self.db.load_torrent_state(h)
            if db_data:
                state.load_from_db(db_data)
                logger.info(f"📦 恢复种子状态: {torrent.name[:20]} (周期#{state.cycle_index})")
            
            state.time_added = time_added
            state.initial_uploaded = total_uploaded
            state.total_size = getattr(torrent, 'total_size', 0) or 0
            
            if state.session_start_time <= 0:
                state.total_uploaded_start = total_uploaded
                state.session_start_time = now
            
            self.states[h] = state
        
        state = self.states[h]
        state.name = torrent.name
        
        if state.total_size <= 0:
            state.total_size = getattr(torrent, 'total_size', 0) or 0
        
        state.speed_tracker.record(now, total_uploaded, total_downloaded, up_speed, dl_speed)
        
        # 检查 peer list
        self._maybe_check_peer_list(state, now)
        
        # 获取属性
        props = self._get_props(h, state, now)
        tl = state.get_tl(now)
        
        if props:
            ra = props.get('reannounce', 0) or 0
            if 0 < ra < C.MAX_REANNOUNCE:
                state.cached_tl = ra
                state.cache_ts = now
                if not state.last_announce_time:
                    tl = ra
        
        current_up_limit = getattr(torrent, 'up_limit', -1) or -1
        
        # 检测周期跳变
        is_jump = state.cycle_start > 0 and tl > state.prev_tl + 30
        
        # 智能等待 TID 搜索
        if not state.monitor_notified:
            self._maybe_search_tid(state, now)
            wait_timeout = (now - state.session_start_time) > 60
            if state.tid_searched or (not self.u2_helper) or wait_timeout:
                self.notifier.monitor_start({
                    'hash': h, 'name': torrent.name, 'total_size': state.total_size,
                    'target': self._get_effective_target(), 'tid': state.tid,
                    'promotion': state.promotion
                })
                state.monitor_notified = True
        
        # 检查下载完成
        progress_val = safe_div(total_downloaded, state.total_size, 0)
        self.notifier.check_finish({
            'hash': h, 'name': torrent.name, 'progress': progress_val,
            'total_uploaded': total_uploaded, 'total_downloaded': total_downloaded
        })
        
        # 新周期处理
        if state.cycle_start == 0 or is_jump:
            if is_jump:
                self._report(state, torrent, now)
            state.new_cycle(now, total_uploaded, tl, is_jump)
            tid_info = f" tid={state.tid}" if state.tid else ""
            sync_status = '✅同步' if state.cycle_synced else '⏳预热'
            logger.info(f"[{torrent.name[:16]}] 🔄 周期 #{state.cycle_index} {sync_status}{tid_info}")
        
        state.prev_tl = tl
        
        # 计算限速
        up_limit, up_reason = self._calc_upload_limit(state, torrent, now, tl)
        dl_limit, dl_reason = self._calc_download_limit(state, torrent, now)
        
        # 检查汇报优化
        self._check_reannounce(state, torrent, now)
        
        # 日志输出
        if now - state.last_log > C.LOG_INTERVAL or state.last_log_limit != up_limit:
            uploaded = state.uploaded_in_cycle(total_uploaded)
            target = self._get_effective_target()
            total = state.estimate_total(now, tl)
            progress = safe_div(uploaded, target * total, 0) * 100
            phase = state.get_phase(now)
            debug = state.last_debug
            pid_out = debug.get('pid_output', 1) if debug else 1
            
            limit_str = 'MAX' if up_limit == -1 else f'{up_limit//1024}K'
            dl_info = f" 📥{dl_limit}K" if dl_limit > 0 else ""
            
            logger.info(f"[{torrent.name[:12]}] ↑{up_speed/1024:.0f}K ({progress:.0f}%) "
                       f"⏱{tl:.0f}s [{phase[0].upper()}] → {limit_str} ({up_reason}) PID={pid_out:.2f}{dl_info}")
            
            state.last_log = now
            state.last_log_limit = up_limit
        
        state.last_up_limit = up_limit
        state.last_up_reason = up_reason
        
        # 应用上传限速
        if up_limit != current_up_limit:
            up_actions.setdefault(up_limit, []).append(h)
            self.modified_up.add(h)
        
        # 应用下载限速
        if dl_limit != state.last_dl_limit:
            if dl_limit > 0:
                state.dl_limited_this_cycle = True
                if state.last_dl_limit <= 0:
                    logger.warning(f"[{torrent.name[:16]}] 📥 下载限速: {dl_limit}K")
                    self.notifier.dl_limit_notify(torrent.name, dl_limit, dl_reason, state.tid)
            elif state.last_dl_limit > 0:
                logger.info(f"[{torrent.name[:16]}] 📥 解除限速")
            
            dl_actions.setdefault(dl_limit * 1024 if dl_limit > 0 else -1, []).append(h)
            self.modified_dl.add(h)
            state.last_dl_limit = dl_limit
        
        return tl
    
    def run(self):
        """主运行循环"""
        cfg = self.config
        target = self._get_effective_target()
        
        logger.info("=" * 60)
        logger.info(f"🚀 qBit Smart Limit v{C.VERSION}")
        logger.info(f"   目标速度: {fmt_speed(target)}")
        logger.info(f"   PT站限制: {fmt_speed(C.SPEED_LIMIT)}")
        logger.info(f"   Tracker: {cfg.target_tracker_keyword or '全部'}")
        logger.info(f"   下载限速: {'✅' if cfg.enable_dl_limit else '❌'}")
        logger.info(f"   汇报优化: {'✅' if cfg.enable_reannounce_opt else '❌'}")
        logger.info(f"   U2辅助: {'✅' if self.u2_enabled else '❌'}")
        logger.info(f"   Telegram: {'✅' if self.notifier.enabled else '❌'}")
        logger.info(f"   数据库: ✅ {C.DB_PATH}")
        logger.info("=" * 60)
        
        self._connect()
        self.notifier.startup(cfg, self.qb_version, self.u2_enabled)
        
        while self.running:
            start = wall_time()
            min_tl = 3600
            
            try:
                self._check_config(start)

                torrents = self.client.torrents_info(status_filter='active')
                
                up_actions: Dict[int, List[str]] = {}
                dl_actions: Dict[int, List[str]] = {}
                now = wall_time()
                
                for t in torrents:
                    if getattr(t, 'state', '') in self.ACTIVE:
                        try:
                            tl = self._process(t, now, up_actions, dl_actions)
                            min_tl = min(min_tl, tl)
                        except Exception as e:
                            logger.debug(f"处理种子异常: {e}")
                
                # 批量应用限速
                for limit, hashes in up_actions.items():
                    try:
                        self.client.torrents_set_upload_limit(limit, hashes)
                    except Exception as e:
                        logger.debug(f"设置上传限速失败: {e}")
                
                for limit, hashes in dl_actions.items():
                    try:
                        self.client.torrents_set_download_limit(limit, hashes)
                    except Exception as e:
                        logger.debug(f"设置下载限速失败: {e}")
                
                # 清理不活跃的种子
                active = {t.hash for t in torrents if getattr(t, 'state', '') in self.ACTIVE}
                for h in list(self.states):
                    if h not in active:
                        del self.states[h]
                
            except APIConnectionError:
                logger.warning("⚠️ 连接断开，重新连接...")
                time.sleep(5)
                try:
                    self._connect()
                except:
                    pass
            except Exception as e:
                logger.error(f"❌ 异常: {e}")
            
            # 动态休眠
            elapsed = wall_time() - start
            if min_tl <= 5:
                sleep = 0.15
            elif min_tl <= 15:
                sleep = 0.25
            elif min_tl <= 30:
                sleep = 0.4
            elif min_tl <= 90:
                sleep = 0.8
            else:
                sleep = 1.5
            
            time.sleep(max(0.1, sleep - elapsed))


# ════════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════════
def main():
    config_paths = [
        "config.json",
        "/etc/qbit-smart-limit/config.json",
        os.path.expanduser("~/.config/qbit-smart-limit/config.json")
    ]
    
    config_path = None
    for p in config_paths:
        if os.path.exists(p):
            config_path = p
            break
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    if not config_path or not os.path.exists(config_path):
        print("❌ 找不到配置文件")
        print("请创建 config.json 或指定配置文件路径")
        print(f"用法: {sys.argv[0]} [config.json]")
        sys.exit(1)
    
    controller = Controller(config_path)
    controller.run()


if __name__ == "__main__":
    main()
