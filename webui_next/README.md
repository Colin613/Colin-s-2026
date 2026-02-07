# 延边朝鲜语语音克隆与TTS系统

基于 Fish Speech 框架构建的专业语音克隆和文本转语音工具，专为延边朝鲜语译制配音设计。

## 功能特性

### 核心功能
- **文本转语音 (TTS)**: 高质量语音生成，支持情感控制、语速调节、音调变化
- **语音克隆**: 基于 30 分钟+音频的专业级语音克隆
- **延边朝鲜语支持**: 针对延边方言优化的语音模型和文本处理
- **批量配音**: SRT 字幕批量生成配音，支持角色声音映射
- **参数控制**: 语速 (0.5x-2.0x)、音调 (±20%)、情感强度、音量增益

### 技术栈
- **后端**: Python + Fish Speech + FastAPI
- **前端**: Next.js 14 + React + Tailwind CSS + shadcn/ui
- **模型**: Fish Audio S1 (支持 LoRA 微调)

## 项目结构

```
fish-speech/
├── fish_speech/                    # 核心模块
│   ├── postprocessing/            # 音频后处理（新增）
│   │   ├── audio_effects.py       # 语速、音调、情感控制
│   │   └── __init__.py
│   ├── text/                      # 文本处理（新增）
│   │   ├── yanbian_normalizer.py  # 延边朝鲜语文本规范化
│   │   └── ...
│   ├── utils/
│   │   └── schema.py              # 扩展的API请求/响应模型
│   └── ...
├── tools/                         # 工具脚本
│   ├── data_preprocessing/        # 数据预处理（新增）
│   │   ├── audio_processor.py     # 音频处理工具
│   │   ├── segmenter.py           # 长音频分段
│   │   ├── transcription_assist.py # 转写辅助
│   │   └── __init__.py
│   ├── batch/                     # 批量处理（新增）
│   │   ├── subtitle_processor.py  # 字幕解析处理
│   │   ├── batch_processor.py     # 批量TTS引擎
│   │   └── __init__.py
│   ├── yanbian_finetune.py        # LoRA微调自动化脚本（新增）
│   ├── server/
│   │   └── views.py               # 扩展的API端点
│   └── ...
└── webui_next/                    # Next.js前端（新增）
    ├── app/
    │   ├── page.tsx               # 首页
    │   ├── tts/page.tsx           # TTS生成页面
    │   ├── voice-studio/page.tsx  # 语音克隆工作室
    │   └── batch-dubbing/page.tsx # 批量配音页面
    ├── components/
    │   └── ui/                    # UI组件
    ├── lib/
    │   ├── api.ts                 # API客户端
    │   └── utils.ts              # 工具函数
    └── ...
```

## 快速开始

### 1. 安装依赖

**后端 (Python)**:
```bash
cd /Users/colin/fish-speech
pip install -r requirements.txt
```

**前端 (Next.js)**:
```bash
cd /Users/colin/fish-speech/webui_next
npm install
```

### 2. 下载模型

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

### 3. 启动服务

**启动后端 API 服务**:
```bash
cd /Users/colin/fish-speech
python tools/api_server.py --listen 127.0.0.1:7860
```

**启动前端 WebUI**:
```bash
cd /Users/colin/fish-speech/webui_next
npm run dev
```

访问:
- WebUI: http://localhost:3000
- API文档: http://localhost:7860/docs

## API 端点

### TTS 生成
```bash
POST /v1/tts
{
  "text": "(happy) 안녕하세요!",
  "speed_factor": 1.0,
  "pitch_factor": 1.0,
  "emotion_intensity": 1.0,
  "volume_gain": 1.0
}
```

### 声音库管理
- `GET /v1/voices/list` - 列出所有声音
- `POST /v1/voices/create` - 创建新声音
- `PUT /v1/voices/{id}` - 更新声音元数据
- `DELETE /v1/voices/{id}` - 删除声音

### 训练任务
- `POST /v1/training/start` - 启动训练
- `GET /v1/training/status/{task_id}` - 查询训练状态
- `POST /v1/training/cancel` - 取消训练

### 批量配音
- `POST /v1/batch/create` - 创建批量任务
- `GET /v1/batch/status/{job_id}` - 查询任务状态
- `GET /v1/batch/list` - 列出所有任务

## 数据准备

### 训练数据格式

```
data/
├── SPK1_YANBIAN/
│   ├── 00.00-05.23.wav
│   ├── 00.00-05.23.lab    # 转写文本
│   └── ...
```

### 自动化训练流程

```bash
python tools/yanbian_finetune.py \
    --data data/yanbian_voice \
    --output checkpoints/yanbian_voice \
    --max-steps 5000
```

## 情感控制

支持的64+种情感标记：

**基本情感**:
- `(angry)` 愤怒
- `(sad)` 悲伤
- `(happy)` 快乐
- `(excited)` 兴奋
- `(surprised)` 惊讶

**高级情感**:
- `(whispering)` 耳语
- `(shouting)` 喊叫
- `(nervous)` 紧张
- `(confident)` 自信

**特殊效果**:
- `(laughter)` 笑声
- `(sighing)` 叹气
- `(gasp)` 倒吸气

示例：
```
(excited) 안녕하세요! (laughter) 반갑습니다!
(sad) 미안해요... (sigh)
```

## 开发环境要求

**后端**:
- Python 3.10+
- CUDA 12.x (GPU)
- RTX 3060/4060 Ti (12GB VRAM) 或更高

**前端**:
- Node.js 18+
- npm 或 yarn

## 许可证

- 代码: Apache 2.0
- 模型权重: CC-BY-NC-SA-4.0

## 参考链接

- [Fish Speech GitHub](https://github.com/fishaudio/fish-speech)
- [Fish Audio 官网](https://fish.audio/)
- [项目计划文档](.claude/plans/squishy-inventing-dolphin.md)

## 贡献

欢迎提交 Issue 和 Pull Request！
