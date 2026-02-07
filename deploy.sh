#!/bin/bash

# Fish Speech 部署助手脚本
# 帮助快速部署到 Vercel 和 HuggingFace Spaces

set -e

echo "========================================"
echo "  Fish Speech 部署助手"
echo "  延边朝鲜语语音克隆系统"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查 Git 仓库
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}⚠️  当前目录不是 Git 仓库${NC}"
    echo ""
    read -p "是否初始化 Git 仓库? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git init
        echo -e "${GREEN}✓ Git 仓库已初始化${NC}"
    else
        echo "请先初始化 Git 仓库后再运行此脚本"
        exit 1
    fi
fi

# 选择部署目标
echo "请选择部署目标:"
echo "  1) 前端 → Vercel"
echo "  2) 后端 → HuggingFace Spaces"
echo "  3) 完整部署 (前端 + 后端)"
echo ""
read -p "请输入选项 (1-3): " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}📦 部署前端到 Vercel...${NC}"
        echo ""
        echo "步骤 1: 推送代码到 GitHub"
        echo "  1. 在 GitHub 创建新仓库"
        echo "  2. 运行以下命令:"
        echo ""
        echo "     git add ."
        echo "     git commit -m \"Initial commit\""
        echo "     git remote add origin https://github.com/YOUR_USERNAME/fish-speech.git"
        echo "     git push -u origin main"
        echo ""
        echo "步骤 2: 在 Vercel 导入项目"
        echo "  1. 访问 https://vercel.com/new"
        echo "  2. 导入你的 GitHub 仓库"
        echo "  3. 配置:"
        echo "     - Root Directory: webui_next"
        echo "     - Build Command: npm run build"
        echo "  4. 添加环境变量:"
        echo "     NEXT_PUBLIC_API_URL = http://localhost:7860"
        echo ""
        echo -e "${GREEN}✓ Vercel 部署指南已显示${NC}"
        ;;

    2)
        echo ""
        echo -e "${BLUE}🐸 部署后端到 HuggingFace Spaces...${NC}"
        echo ""
        echo "步骤 1: 在 HuggingFace 创建 Space"
        echo "  1. 访问 https://huggingface.co/spaces"
        echo "  2. 点击 'Create new Space'"
        echo "  3. 配置:"
        echo "     - Space name: fish-speech-api"
        echo "     - SDK: Docker"
        echo "     - Hardware: CPU basic"
        echo ""
        echo "步骤 2: 准备 Dockerfile"
        echo "  在 Space 中创建以下文件:"
        echo ""
        echo "  1. 复制 Dockerfile.hf 内容到 Dockerfile"
        echo "  2. 复制 requirements-hf.txt 内容到 requirements.txt"
        echo "  3. 复制 README_SPACES.md 内容到 README.md"
        echo ""
        echo "步骤 3: 上传文件"
        echo "  方式 1: 使用 Git (推荐)"
        echo "     git clone https://huggingface.co/spaces/YOUR_USERNAME/fish-speech-api"
        echo "     cp -r . fish-speech-api/"
        echo "     cd fish-speech-api"
        echo "     git add ."
        echo "     git commit -m \"Initial deployment\""
        echo "     git push"
        echo ""
        echo "  方式 2: 直接在网页上编辑文件"
        echo ""
        echo -e "${GREEN}✓ HuggingFace Spaces 部署指南已显示${NC}"
        ;;

    3)
        echo ""
        echo -e "${BLUE}🚀 完整部署指南...${NC}"
        echo ""
        echo "=== 前端部署 (Vercel) ==="
        echo ""
        echo "1. 推送代码到 GitHub:"
        echo "   git add ."
        echo "   git commit -m \"Deploy Fish Speech\""
        echo "   git remote add origin https://github.com/YOUR_USERNAME/fish-speech.git"
        echo "   git push -u origin main"
        echo ""
        echo "2. 在 https://vercel.com/new 导入项目"
        echo "   Root Directory: webui_next"
        echo ""
        echo "3. 配置环境变量 (Vercel):"
        echo "   NEXT_PUBLIC_API_URL = https://YOUR_USERNAME-fish-speech-api.hf.space"
        echo ""
        echo ""
        echo "=== 后端部署 (HuggingFace Spaces) ==="
        echo ""
        echo "1. 创建 Space: https://huggingface.co/spaces"
        echo "   - SDK: Docker"
        echo "   - Visibility: Public"
        echo ""
        echo "2. 克隆 Space 并上传文件:"
        echo "   git clone https://huggingface.co/spaces/YOUR_USERNAME/fish-speech-api"
        echo "   cd fish-speech-api"
        echo "   # 复制以下文件到当前目录:"
        echo "   # - Dockerfile.hf → Dockerfile"
        echo "   # - requirements-hf.txt → requirements.txt"
        echo "   # - README_SPACES.md → README.md"
        echo "   # - 整个 fish_speech 目录"
        echo "   # - 整个 tools 目录"
        echo "   git add ."
        echo "   git push"
        echo ""
        echo "3. 等待构建完成 (约 5-10 分钟)"
        echo ""
        echo "4. 获取后端 URL:"
        echo "   https://YOUR_USERNAME-fish-speech-api.hf.space"
        echo ""
        echo ""
        echo -e "${GREEN}✓ 完整部署指南已显示${NC}"
        echo ""
        echo -e "${YELLOW}📝 重要提示:${NC}"
        echo "  1. 后端部署完成后，更新 Vercel 的环境变量"
        echo "  2. HuggingFace Spaces 首次启动需要下载模型，请耐心等待"
        echo "  3. 详细部署文档请查看 DEPLOYMENT.md"
        ;;

    *)
        echo "无效的选项"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  部署指南已显示完成${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "如需更多帮助，请查看 DEPLOYMENT.md 文档"
