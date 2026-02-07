# 延边朝鲜语语音克隆系统 - 部署指南

本文档介绍如何将延边朝鲜语语音克隆系统部署到生产环境。

## 系统架构

系统由两部分组成：
- **前端**: Next.js 14 应用，部署到 Vercel
- **后端**: Python FastAPI 应用，部署到支持 Python 的平台

---

## 部署方案

### 方案 A: 前端 Vercel + 后端 HuggingFace Spaces (推荐)

这是最简单且免费的方案。

#### 1. 前端部署到 Vercel

**前提条件**:
- GitHub 账号
- Vercel 账号 (使用 GitHub 登录)

**步骤**:

1. 将代码推送到 GitHub 仓库
```bash
cd /Users/colin/fish-speech
git init
git add .
git commit -m "Initial commit: Yanbian Korean voice cloning system"
git remote add origin https://github.com/YOUR_USERNAME/fish-speech.git
git push -u origin main
```

2. 在 Vercel 导入项目
   - 访问 https://vercel.com/new
   - 导入你的 GitHub 仓库
   - 配置项目：
     - Framework Preset: Next.js
     - Root Directory: `webui_next`
     - Build Command: `npm run build`
     - Output Directory: `.next`

3. 配置环境变量
   - 在 Vercel 项目设置中添加：
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.hf.space
   ```

4. 部署完成
   - Vercel 会自动部署
   - 访问分配的 URL (如 `https://fish-speech.vercel.app`)

#### 2. 后端部署到 HuggingFace Spaces

**前提条件**:
- HuggingFace 账号

**步骤**:

1. 创建新的 Space
   - 访问 https://huggingface.co/spaces
   - 点击 "Create new Space"
   - 配置：
     - Owner: 你的用户名
     - Space name: `fish-speech-api`
     - License: MIT
     - SDK: Docker
     - Hardware: CPU basic (免费) 或 GPU (付费)

2. 创建文件结构

在 Space 中创建以下文件：

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 克隆仓库（或复制文件）
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "tools/api_server.py", "--host", "0.0.0.0", "--port", "7860"]
```

**requirements.txt** (确保包含所有依赖):
```
fastapi
uvicorn
kui
loguru
torch
torchaudio
numpy
click
hydra-core
omegaconf
protobuf
grpcio-tools
pydantic
```

**README.md**:
```markdown
---
title: Fish Speech API
emoji: 🐟
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Fish Speech API

延边朝鲜语语音克隆 API 服务
```

3. 上传文件到 Space
   - 使用 Git 上传或直接在网页上编辑文件
   - HuggingFace 会自动构建和部署

4. 获取后端 URL
   - 部署完成后，URL 类似：`https://your-username-fish-speech-api.hf.space`

#### 3. 配置 CORS

更新后端以允许来自 Vercel 的跨域请求。

在 `tools/server/views.py` 中添加 CORS 中间件：

```python
from fastapi.middleware.cors import CORSMiddleware
from kui import Kui

# 在 app 创建后添加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 4. 更新前端环境变量

在 Vercel 项目设置中更新：
```
NEXT_PUBLIC_API_URL=https://your-username-fish-speech-api.hf.space
```

重新部署前端（推送新代码或使用 Vercel 控制面板）。

---

### 方案 B: 使用 Railway (推荐用于 GPU)

Railway 支持 Python 且提供 GPU 实例。

1. 访问 https://railway.app
2. 连接 GitHub 仓库
3. 选择 `fish-speech` 项目
4. 配置：
   - Build Command: 留空（自动检测）
   - Start Command: `python tools/api_server.py --host 0.0.0.0 --port $PORT`
5. 添加环境变量：
   - `PORT`: 7860
   - `PYTHON_VERSION`: 3.10
6. 部署后获取 URL

---

## 环境变量配置

### 前端 (.env.local 或 Vercel 环境变量)
```bash
NEXT_PUBLIC_API_URL=http://localhost:7860  # 本地开发
# NEXT_PUBLIC_API_URL=https://your-backend.hf.space  # 生产环境
```

### 后端 (可选)
```bash
CUDA_VISIBLE_DEVICES=0  # GPU 设备
MODEL_PATH=/app/checkpoints  # 模型路径
```

---

## 生产环境检查清单

### 前端 (Vercel)
- [ ] 代码推送到 GitHub
- [ ] Vercel 项目导入成功
- [ ] NEXT_PUBLIC_API_URL 配置正确
- [ ] 构建成功无错误
- [ ] 部署后页面可访问
- [ ] API 请求正常发送

### 后端 (HuggingFace Spaces)
- [ ] Dockerfile 配置正确
- [ ] requirements.txt 包含所有依赖
- [ ] Space 构建成功
- [ ] 服务正常运行（查看日志）
- [ ] API 端点可访问
- [ ] CORS 配置正确

### 功能测试
- [ ] TTS 生成正常
- [ ] 语音上传成功
- [ ] 语音克隆训练启动
- [ ] 声音库显示正常
- [ ] 音频播放功能正常

---

## 故障排查

### 前端问题

**问题**: API 请求失败 (CORS 错误)
- 检查后端 CORS 配置
- 确认 NEXT_PUBLIC_API_URL 正确

**问题**: 页面无法加载
- 检查 Vercel 部署日志
- 确认 Next.js 构建成功

### 后端问题

**问题**: Space 构建失败
- 检查 Dockerfile 语法
- 查看 Space 的构建日志
- 确认 requirements.txt 依赖完整

**问题**: API 请求超时
- HuggingFace Spaces CPU 基础版可能有冷启动
- 考虑升级到 GPU 或使用其他平台

**问题**: 训练内存不足
- GPU 基础版可能内存不够
- 需要至少 8GB VRAM 进行 LoRA 训练
- 考虑使用 Colab 或本地 GPU 进行训练

---

## 本地开发

在部署前，建议先在本地测试完整功能：

1. 启动后端:
```bash
cd /Users/colin/fish-speech
python tools/api_server.py --host 0.0.0.0 --port 7860
```

2. 启动前端:
```bash
cd webui_next
npm run dev
```

3. 访问 http://localhost:3000

---

## 费用估算

### Vercel (前端)
- 免费套餐: 100GB 带宽/月
- 付费套餐: $20/月起

### HuggingFace Spaces (后端)
- CPU 基础版: 免费
- GPU (T4): $0.10/小时
- GPU (A10G): $1.00/小时

### Railway (后端)
- 免费套餐: $5/月额度
- GPU 实例: 按使用量计费

---

## 联系与支持

如有问题，请查看：
- Fish Speech 官方文档
- Vercel 部署文档
- HuggingFace Spaces 文档
