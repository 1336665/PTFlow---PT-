# PTFlow - PT刷流管理系统

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-green.svg" alt="version">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license">
  <img src="https://img.shields.io/badge/python-3.11+-yellow.svg" alt="python">
  <img src="https://img.shields.io/badge/node-20+-orange.svg" alt="node">
</p>

一个现代化的PT刷流管理系统，支持多qBittorrent实例管理、智能动态限速、RSS自动订阅等功能。

## ✨ 功能特性

### 🖥️ 多实例管理
- 支持同时管理多个 qBittorrent 客户端
- 实时监控各实例状态和速度
- 统一界面管理所有种子

### 🎯 智能动态限速
- 基于 PID 控制器和卡尔曼滤波的精准限速算法
- 四阶段限速策略（预热/追赶/稳定/收尾）
- 自动适应汇报周期，精准控制上传速度
- 支持按站点独立配置限速规则

### 📡 RSS自动订阅
- 支持多个RSS源订阅
- 灵活的过滤规则（包含/排除关键词）
- 自动下载新种子到指定实例
- 可关联站点配置自动应用限速规则

### 📱 响应式界面
- 现代化深色主题设计
- 完美支持移动端访问
- 实时WebSocket数据更新

## 🚀 快速开始

### 一键安装（推荐）

```bash
# 下载并执行安装脚本
curl -fsSL https://raw.githubusercontent.com/yourname/ptflow/main/scripts/install.sh | bash
```

### Docker 安装

```bash
# 克隆项目
git clone https://github.com/yourname/ptflow.git
cd ptflow

# 启动服务
docker-compose up -d
```

### 手动安装

#### 后端

```bash
cd backend
pip install -r requirements.txt
python main.py
```

#### 前端

```bash
cd frontend
npm install
npm run dev      # 开发模式
npm run build    # 构建生产版本
```

## 📖 使用指南

### 1. 登录系统

默认账号：`admin` / `admin`

⚠️ **首次登录后请立即修改密码！**

### 2. 添加qB实例

1. 进入「qB实例」页面
2. 点击「添加实例」
3. 填写 qBittorrent WebUI 信息
4. 测试连接并保存

### 3. 配置站点限速

1. 进入「站点配置」页面
2. 点击「添加站点」
3. 填写站点名称和 Tracker 关键词
4. 设置目标速度和安全边际
5. 启用需要的功能（动态限速/下载限速/汇报优化）

### 4. 添加RSS订阅

1. 进入「RSS订阅」页面
2. 点击「添加订阅」
3. 填写RSS地址和下载设置
4. 可选配置过滤规则
5. 启用自动下载

## 🛠️ 配置说明

### 限速参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| 目标速度 | 站点限速标准 | 按站点规则设置 |
| 安全边际 | 实际目标 = 目标速度 × 安全边际 | 0.95 ~ 0.99 |
| 动态限速 | 启用智能限速算法 | 开启 |
| 下载限速 | 上传超速时限制下载 | 开启 |
| 汇报优化 | 智能触发强制汇报 | 开启 |

### 限速阶段说明

| 阶段 | 触发条件 | 策略 |
|------|----------|------|
| 🔥 预热 | 未同步周期 | 保守策略，建立基准 |
| 🏃 追赶 | 距汇报 > 120s | 如落后则放宽限速 |
| ⚖️ 稳定 | 30s ~ 120s | 精确控制 |
| 🎯 收尾 | < 30s | 最高精度微调 |

## 📡 API文档

启动服务后访问 `http://localhost:8000/docs` 查看完整API文档。

### 主要接口

```
POST /api/auth/login          # 登录
GET  /api/qb/instances        # 获取qB实例列表
GET  /api/qb/instances/{id}/torrents  # 获取种子列表
GET  /api/sites               # 获取站点配置
GET  /api/rss                 # 获取RSS订阅
GET  /api/limit/states        # 获取限速状态
GET  /api/stats               # 获取统计信息
```

## 🔧 常见问题

### Q: 无法连接qBittorrent?

1. 确保 qBittorrent WebUI 已启用
2. 检查 WebUI 地址和端口是否正确
3. 确认用户名密码无误
4. 如果使用Docker，注意使用正确的网络地址（如 `host.docker.internal`）

### Q: 限速不生效?

1. 检查站点配置的 Tracker 关键词是否正确匹配
2. 确认「动态限速」功能已启用
3. 查看「动态限速」页面确认引擎状态
4. 检查日志排查问题

### Q: RSS不自动下载?

1. 确认RSS地址可正常访问
2. 检查过滤规则是否过于严格
3. 确认「自动下载」已启用
4. 查看RSS预览确认是否有新条目

### Q: 如何更新版本?

```bash
cd /opt/ptflow
docker-compose pull
docker-compose up -d
```

### Q: 如何查看日志?

```bash
cd /opt/ptflow
docker-compose logs -f
```

### Q: 如何备份数据?

数据文件位于 Docker volume `ptflow_ptflow_data` 中，可通过以下命令备份：

```bash
docker run --rm -v ptflow_ptflow_data:/data -v $(pwd):/backup alpine tar czf /backup/ptflow-backup.tar.gz /data
```

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend                           │
│         React + TailwindCSS + Zustand                   │
├─────────────────────────────────────────────────────────┤
│                      Backend                            │
│              FastAPI + SQLite + JWT                     │
├────────────────┬────────────────┬───────────────────────┤
│  qB Manager    │  Limit Engine  │    RSS Manager        │
│  多实例管理    │   动态限速     │     RSS订阅           │
├────────────────┴────────────────┴───────────────────────┤
│                   qBittorrent API                       │
└─────────────────────────────────────────────────────────┘
```

### 限速算法

动态限速采用 **PID + 卡尔曼滤波** 组合算法：

1. **卡尔曼滤波器**：预测上传速度趋势
2. **PID控制器**：根据目标偏差调整限速
3. **自适应量化**：根据阶段调整精度
4. **平滑过渡**：避免限速值剧烈波动

## 📁 项目结构

```
ptflow/
├── backend/                 # 后端代码
│   ├── main.py             # 主程序
│   └── requirements.txt    # Python依赖
├── frontend/               # 前端代码
│   ├── src/
│   │   ├── components/     # 组件
│   │   ├── pages/          # 页面
│   │   ├── store.js        # 状态管理
│   │   └── utils/          # 工具函数
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── install.sh          # 一键安装脚本
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [qBittorrent](https://www.qbittorrent.org/) - 优秀的BT客户端
- [FastAPI](https://fastapi.tiangolo.com/) - 现代化Python Web框架
- [React](https://react.dev/) - 前端框架
- [TailwindCSS](https://tailwindcss.com/) - CSS框架
- [Vertex](https://github.com/vertex-app/vertex) - 项目参考

## ⭐ Star History

如果这个项目对你有帮助，请给一个 Star ⭐️

---

<p align="center">Made with ❤️ for PT Community</p>
