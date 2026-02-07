---
title: Fish Speech API
emoji: 🐟
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
---

# Fish Speech API - 延边朝鲜语语音克隆系统

## 📖 简介

这是 Fish Speech 的 API 服务，专门用于延边朝鲜语语音克隆和文本转语音（TTS）。

## 🚀 功能

- **语音克隆**: 上传音频文件进行 LoRA 训练，实现高精度语音克隆（90-95% 相似度）
- **TTS 生成**: 支持中文、朝鲜语等多种语言的语音合成
- **声音库管理**: 管理多个克隆声音
- **参数控制**: 支持语速、音调、情感强度调节

## 📡 API 端点

### 健康检查
```
GET /v1/health
```

### TTS 生成
```
POST /v1/tts
Content-Type: application/json

{
  "text": "你好，这是一个测试。",
  "reference_id": "voice_id_here"  // 可选，使用声音库中的声音
}
```

### 语音克隆
```
POST /v1/voice/clone
Content-Type: multipart/form-data

- audio: 音频文件
- name: 声音名称
- reference_text: 参考文本
```

### 训练状态
```
GET /v1/training/status/{task_id}
```

### 声音库列表
```
GET /v1/voices/list
```

## ⚙️ 配置说明

- **端口**: 7860
- **Worker 数量**: 1（HuggingFace Spaces CPU 基础版限制）
- **超时时间**: 训练任务可能需要 30-60 分钟

## 💡 使用提示

1. **首次请求可能较慢**: 模型需要加载到内存
2. **训练需要 GPU**: CPU 版本训练非常慢，建议使用 GPU 或本地训练
3. **音频要求**: 建议上传 30 秒以上的高质量音频

## 📚 相关链接

- [前端部署](https://your-vercel-app-url.vercel.app)
- [GitHub 仓库](https://github.com/YOUR_USERNAME/fish-speech)
- [Fish Speech 文档](https://github.com/fishaudio/fish-speech)

## 📝 许可证

Apache License 2.0
