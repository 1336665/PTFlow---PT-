#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════════
# PTFlow 一键安装脚本
# PT刷流管理系统
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 版本
VERSION="1.0.0"

# 打印带颜色的消息
print_msg() {
    echo -e "${GREEN}[PTFlow]${NC} $1"
}

print_error() {
    echo -e "${RED}[错误]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[信息]${NC} $1"
}

# 打印 Banner
print_banner() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║     ██████╗ ████████╗███████╗██╗      ██████╗ ██╗    ██╗     ║"
    echo "║     ██╔══██╗╚══██╔══╝██╔════╝██║     ██╔═══██╗██║    ██║     ║"
    echo "║     ██████╔╝   ██║   █████╗  ██║     ██║   ██║██║ █╗ ██║     ║"
    echo "║     ██╔═══╝    ██║   ██╔══╝  ██║     ██║   ██║██║███╗██║     ║"
    echo "║     ██║        ██║   ██║     ███████╗╚██████╔╝╚███╔███╔╝     ║"
    echo "║     ╚═╝        ╚═╝   ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝      ║"
    echo "║                                                               ║"
    echo "║                   PT刷流管理系统 v${VERSION}                      ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 检查是否为 root 用户
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "请使用 root 用户运行此脚本"
        exit 1
    fi
}

# 检查系统
check_system() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION_ID=$VERSION_ID
    else
        print_error "无法识别操作系统"
        exit 1
    fi
    
    print_info "检测到系统: $OS $VERSION_ID"
}

# 安装 Docker
install_docker() {
    if command -v docker &> /dev/null; then
        print_msg "Docker 已安装"
        return
    fi
    
    print_info "正在安装 Docker..."
    
    case $OS in
        ubuntu|debian)
            apt-get update
            apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
            curl -fsSL https://download.docker.com/linux/$OS/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/$OS $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
            apt-get update
            apt-get install -y docker-ce docker-ce-cli containerd.io
            ;;
        centos|rhel|fedora)
            yum install -y yum-utils
            yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            yum install -y docker-ce docker-ce-cli containerd.io
            ;;
        *)
            print_error "不支持的操作系统: $OS"
            print_info "请手动安装 Docker: https://docs.docker.com/engine/install/"
            exit 1
            ;;
    esac
    
    systemctl enable docker
    systemctl start docker
    print_msg "Docker 安装完成"
}

# 安装 Docker Compose
install_docker_compose() {
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_msg "Docker Compose 已安装"
        return
    fi
    
    print_info "正在安装 Docker Compose..."
    
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d'"' -f4)
    curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    
    print_msg "Docker Compose 安装完成"
}

# 创建安装目录
create_install_dir() {
    INSTALL_DIR="/opt/ptflow"
    
    if [ -d "$INSTALL_DIR" ]; then
        print_warn "安装目录已存在: $INSTALL_DIR"
        read -p "是否覆盖安装? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "安装已取消"
            exit 0
        fi
    fi
    
    mkdir -p "$INSTALL_DIR"
    print_msg "创建安装目录: $INSTALL_DIR"
}

# 下载项目文件
download_project() {
    print_info "正在下载 PTFlow..."
    
    cd "$INSTALL_DIR"
    
    # 如果有 git，使用 git clone
    if command -v git &> /dev/null; then
        if [ -d ".git" ]; then
            git pull
        else
            git clone https://github.com/yourname/ptflow.git . 2>/dev/null || {
                print_warn "Git clone 失败，尝试下载 release..."
                download_release
            }
        fi
    else
        download_release
    fi
}

# 下载 release 包
download_release() {
    print_info "正在下载 release 包..."
    
    # 这里应该是实际的下载地址
    RELEASE_URL="https://github.com/yourname/ptflow/releases/latest/download/ptflow.tar.gz"
    
    curl -L "$RELEASE_URL" -o ptflow.tar.gz 2>/dev/null || {
        print_error "下载失败，请检查网络连接"
        print_info "您也可以手动下载项目到 $INSTALL_DIR"
        exit 1
    }
    
    tar -xzf ptflow.tar.gz
    rm ptflow.tar.gz
}

# 配置环境变量
configure_env() {
    print_info "配置环境变量..."
    
    # 生成随机 JWT Secret
    JWT_SECRET=$(openssl rand -hex 32 2>/dev/null || cat /dev/urandom | tr -dc 'a-zA-Z0-9' | head -c 64)
    
    # 创建 .env 文件
    cat > "$INSTALL_DIR/.env" << EOF
# PTFlow 环境配置
JWT_SECRET=$JWT_SECRET
TZ=Asia/Shanghai
EOF
    
    print_msg "环境变量配置完成"
}

# 启动服务
start_service() {
    print_info "正在启动 PTFlow..."
    
    cd "$INSTALL_DIR"
    
    # 使用 docker compose 或 docker-compose
    if docker compose version &> /dev/null; then
        docker compose up -d --build
    else
        docker-compose up -d --build
    fi
    
    print_msg "PTFlow 启动成功!"
}

# 创建系统服务
create_systemd_service() {
    print_info "创建系统服务..."
    
    cat > /etc/systemd/system/ptflow.service << EOF
[Unit]
Description=PTFlow - PT刷流管理系统
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable ptflow
    
    print_msg "系统服务创建完成"
}

# 显示完成信息
show_complete() {
    # 获取服务器 IP
    SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                    PTFlow 安装完成!                           ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}访问地址:${NC} http://${SERVER_IP}:8000"
    echo ""
    echo -e "  ${CYAN}默认账号:${NC} admin"
    echo -e "  ${CYAN}默认密码:${NC} admin"
    echo ""
    echo -e "  ${YELLOW}请登录后立即修改密码!${NC}"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  常用命令:"
    echo -e "    ${PURPLE}查看状态:${NC} cd $INSTALL_DIR && docker compose ps"
    echo -e "    ${PURPLE}查看日志:${NC} cd $INSTALL_DIR && docker compose logs -f"
    echo -e "    ${PURPLE}重启服务:${NC} cd $INSTALL_DIR && docker compose restart"
    echo -e "    ${PURPLE}停止服务:${NC} cd $INSTALL_DIR && docker compose down"
    echo -e "    ${PURPLE}更新版本:${NC} cd $INSTALL_DIR && docker compose pull && docker compose up -d"
    echo ""
}

# 卸载函数
uninstall() {
    print_banner
    print_warn "即将卸载 PTFlow..."
    
    read -p "是否同时删除数据? (y/n): " -n 1 -r
    echo
    
    if [ -d "/opt/ptflow" ]; then
        cd /opt/ptflow
        docker compose down 2>/dev/null || true
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker volume rm ptflow_ptflow_data 2>/dev/null || true
        fi
        
        rm -rf /opt/ptflow
    fi
    
    systemctl stop ptflow 2>/dev/null || true
    systemctl disable ptflow 2>/dev/null || true
    rm -f /etc/systemd/system/ptflow.service
    systemctl daemon-reload
    
    print_msg "PTFlow 已卸载"
}

# 主函数
main() {
    print_banner
    
    case "${1:-}" in
        uninstall|remove)
            check_root
            uninstall
            exit 0
            ;;
        *)
            ;;
    esac
    
    check_root
    check_system
    install_docker
    install_docker_compose
    create_install_dir
    # download_project  # 如果是从 GitHub 安装，取消注释
    configure_env
    start_service
    create_systemd_service
    show_complete
}

# 运行主函数
main "$@"
