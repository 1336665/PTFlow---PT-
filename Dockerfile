# PTFlow Dockerfile
# 多阶段构建

# 阶段1: 构建前端
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# 阶段2: 运行后端
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY backend/ ./

# 从前端构建阶段复制静态文件
COPY --from=frontend-builder /app/backend/static ./static

# 创建数据目录
RUN mkdir -p /data

# 环境变量
ENV DATABASE_PATH=/data/ptflow.db
ENV JWT_SECRET=""
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/stats || exit 1

# 启动命令
CMD ["python", "main.py"]
