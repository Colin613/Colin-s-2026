"use client";

import { useState, useEffect, useRef } from "react";

// API base URL - configurable via environment variable
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

export default function IntegratedAppPage() {
  const [activeTab, setActiveTab] = useState("tts");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // TTS State
  const [text, setText] = useState("");
  const [speedFactor, setSpeedFactor] = useState(1.0);
  const [pitchFactor, setPitchFactor] = useState(1.0);
  const [emotion, setEmotion] = useState("");
  const [generating, setGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [selectedVoiceForTTS, setSelectedVoiceForTTS] = useState<string>(""); // Selected voice from library

  // Voice Clone State
  const [voiceName, setVoiceName] = useState("");
  const [voiceDescription, setVoiceDescription] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null); // Audio preview URL
  const [referenceText, setReferenceText] = useState("请输入音频中实际说的话"); // Reference audio content text
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [trainingMaxSteps, setTrainingMaxSteps] = useState(5000);
  const [trainingLearningRate, setTrainingLearningRate] = useState(1e-4);
  const [trainingBatchSize, setTrainingBatchSize] = useState(16);

  // Data State
  const [voices, setVoices] = useState<any[]>([]);
  const [trainingTasks, setTrainingTasks] = useState<any[]>([]);
  const [batchJobs, setBatchJobs] = useState<any[]>([]);

  // Voice Test State
  const [testingVoice, setTestingVoice] = useState<string | null>(null);
  const [testAudioUrl, setTestAudioUrl] = useState<string | null>(null);
  const [voiceTestText, setVoiceTestText] = useState("안녕하세요? 이것은 테스트입니다.");
  const [voiceTestEmotion, setVoiceTestEmotion] = useState("");
  const [voiceTestSpeed, setVoiceTestSpeed] = useState(1.0);
  const [voiceTestPitch, setVoiceTestPitch] = useState(1.0);
  const [selectedVoiceForTest, setSelectedVoiceForTest] = useState<any>(null);

  // Ref to track if the current test request is still valid (for avoiding race conditions)
  const currentTestVoiceRef = useRef<string | null>(null);

  // Emotions for TTS
  const emotions = [
    { value: "", label: "默认" },
    { value: "(angry)", label: "愤怒" },
    { value: "(sad)", label: "悲伤" },
    { value: "(happy)", label: "快乐" },
    { value: "(excited)", label: "兴奋" },
    { value: "(surprised)", label: "惊讶" },
  ];

  // Generate TTS
  const handleGenerate = async () => {
    if (!text.trim()) return;
    setGenerating(true);

    try {
      const textWithEmotion = emotion ? `${emotion} ${text}` : text;

      // Use voice-specific endpoint if a voice is selected, otherwise use default TTS
      const endpoint = selectedVoiceForTTS ? `${API_BASE}/v1/voice/tts` : `${API_BASE}/v1/tts`;

      const body: any = {
        text: textWithEmotion,
        speed_factor: speedFactor,
        pitch_factor: pitchFactor,
        format: "wav",
      };

      // Add voice_id if using voice-specific TTS
      if (selectedVoiceForTTS) {
        body.voice_id = selectedVoiceForTTS;
        // Map emotion to the format expected by voice TTS
        if (emotion) {
          const emotionMap: Record<string, string> = {
            "(angry)": "angry",
            "(sad)": "sad",
            "(happy)": "happy",
            "(excited)": "happy",
            "(surprised)": "happy",
          };
          body.emotion = emotionMap[emotion] || "";
        }
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
      } else {
        const error = await response.json();
        alert(`生成失败: ${error.detail || error.message || "未知错误"}`);
      }
    } catch (error) {
      console.error("TTS error:", error);
      alert(`生成失败: 请确保后端服务运行在 http://localhost:7860`);
    } finally {
      setGenerating(false);
    }
  };

  // Voice Clone - Upload & Start Training
  const handleVoiceClone = async () => {
    if (!voiceName.trim() || !audioFile) {
      alert("请填写声音名称并上传音频文件");
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      // Create form data
      const formData = new FormData();
      formData.append("audio", audioFile);
      formData.append("name", voiceName);
      formData.append("description", voiceDescription);
      formData.append("reference_text", referenceText.trim() || "请输入音频中实际说的话"); // Reference audio content
      formData.append("max_steps", trainingMaxSteps.toString());
      formData.append("learning_rate", trainingLearningRate.toString());
      formData.append("batch_size", trainingBatchSize.toString());

      // Upload with progress
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const percentComplete = (e.loaded / e.total) * 100;
          setUploadProgress(percentComplete);
        }
      });

      xhr.addEventListener("load", () => {
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          alert(`✅ LoRA 训练任务已创建!\n\n任务ID: ${response.task_id}\n\n⏱️ 预计训练时间: ${response.estimated_time_minutes || 60} 分钟\n🎯 训练完成后相似度: 90-95%\n\n请切换到「模型训练」标签页查看进度`);
          // Reset form
          setVoiceName("");
          setVoiceDescription("");
          setReferenceText("请输入音频中实际说的话");
          setAudioFile(null);
          // Clean up preview URL
          if (audioPreviewUrl) {
            URL.revokeObjectURL(audioPreviewUrl);
            setAudioPreviewUrl(null);
          }
          setUploadProgress(0);
          // Reload data
          loadData();
          // Switch to training tab to see progress
          setActiveTab("training");
        } else {
          const error = JSON.parse(xhr.responseText);
          alert(`上传失败: ${error.detail || "未知错误"}`);
        }
        setUploading(false);
      });

      xhr.addEventListener("error", () => {
        alert("上传失败: 请确保后端服务运行在 http://localhost:7860");
        setUploading(false);
      });

      xhr.open("POST", `${API_BASE}/v1/voice-clone/create`);
      xhr.send(formData);
    } catch (error) {
      console.error("Clone error:", error);
      alert(`上传失败: 请确保后端服务运行在 http://localhost:7860`);
      setUploading(false);
    }
  };

  // Test voice (generate TTS with selected voice)
  const handleTestVoice = async (voiceId: string, voice: any) => {
    // If already testing this voice, don't restart
    if (testingVoice === voiceId) return;

    // If clicking the same voice while audio is already loaded, just play it
    if (testAudioUrl && selectedVoiceForTest?.id === voice.id && testingVoice !== voiceId) {
      const audio = document.querySelector(`audio[data-voice-id="${voiceId}"]`);
      if (audio) {
        audio.currentTime = 0;
        audio.play();
      }
      return;
    }

    setTestingVoice(voiceId);
    setSelectedVoiceForTest(voice);
    setTestAudioUrl(null);

    try {
      // Use custom test text if set, otherwise use Korean default
      const testText = voiceTestText.trim() || "안녕하세요? 이것은 테스트입니다.";

      // Set the ref to track this request
      currentTestVoiceRef.current = voiceId;

      // Use voice-specific TTS endpoint with new parameters
      const response = await fetch(`${API_BASE}/v1/voice/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          voice_id: voiceId,
          text: testText,
          format: "wav",
          emotion: voiceTestEmotion,
          speed: voiceTestSpeed,
          pitch: voiceTestPitch,
        }),
      });

      // Check if this is still the current request (prevent race conditions)
      if (currentTestVoiceRef.current !== voiceId) {
        return; // Request was superseded by a new one
      }

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setTestAudioUrl(url);
        // Clear testing state immediately when audio is ready
        setTestingVoice(null);
      } else {
        const error = await response.json().catch(() => ({ message: "未知错误" }));
        alert(`生成失败: ${error.message || error.detail || "未知错误"}`);
        setTestingVoice(null);
      }
    } catch (error) {
      console.error("Test voice error:", error);
      alert(`生成失败: 请确保后端服务运行在 http://localhost:7860`);
      setTestingVoice(null);
    }
  };

  // Delete voice
  const handleDeleteVoice = async (voiceId: string, voiceName: string) => {
    if (!confirm(`确定要删除声音 "${voiceName}" 吗？此操作无法撤销。`)) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/v1/voices/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: voiceId }),
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success) {
          alert("删除成功");
          // Clear selected voice if it was the deleted one
          if (selectedVoiceForTest?.id === voiceId) {
            setSelectedVoiceForTest(null);
            setTestAudioUrl(null);
          }
          loadData();
        } else {
          alert(`删除失败: ${result.message || "未知错误"}`);
        }
      } else {
        const error = await response.json().catch(() => ({ message: "未知错误" }));
        alert(`删除失败: ${error.message || error.detail || "未知错误"}`);
      }
    } catch (error) {
      console.error("Delete voice error:", error);
      alert(`删除失败: 请确保后端服务运行在 http://localhost:7860`);
    }
  };

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Check file size (30 minutes at 24kHz mono ≈ 25MB, allow up to 500MB)
      if (file.size > 500 * 1024 * 1024) {
        alert("文件太大! 请上传小于 500MB 的音频文件");
        return;
      }

      // Clean up old URL if exists
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }

      // Create preview URL for the audio file
      const url = URL.createObjectURL(file);
      setAudioPreviewUrl(url);
      setAudioFile(file);
    }
  };

  // Format file size
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  // Load data
  const loadData = async () => {
    try {
      const headers = { "Accept": "application/json" };

      const [voicesRes, tasksRes, jobsRes] = await Promise.all([
        fetch(`${API_BASE}/v1/voices/list`, { headers }).catch(() => null),
        fetch(`${API_BASE}/v1/training/list`, { headers }).catch(() => null),
        fetch(`${API_BASE}/v1/batch/list`, { headers }).catch(() => null),
      ]);

      if (voicesRes?.ok) {
        const data = await voicesRes.json();
        setVoices(data.voices || []);
      }
      if (tasksRes?.ok) {
        const data = await tasksRes.json();
        setTrainingTasks(data.tasks || []);
      }
      if (jobsRes?.ok) {
        const data = await jobsRes.json();
        setBatchJobs(data.jobs || []);
      }
    } catch (error) {
      console.error("Load data error:", error);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Auto-refresh training tasks when training tab is active
  useEffect(() => {
    if (activeTab !== "training") return;

    // Check if there are any running tasks
    const hasRunningTasks = trainingTasks.some(
      (task) => task.status === "running" || task.status === "training" ||
                task.status === "preparing_data" || task.status === "extracting_vq" ||
                task.status === "building_dataset" || task.status === "merging_weights"
    );

    if (!hasRunningTasks) return;

    // Set up auto-refresh every 5 seconds
    const interval = setInterval(() => {
      loadData();
    }, 5000);

    return () => clearInterval(interval);
  }, [activeTab, trainingTasks]);

  // Cleanup audio preview URL on unmount
  useEffect(() => {
    return () => {
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }
    };
  }, [audioPreviewUrl]);

  // Navigation
  const navItems = [
    { id: "tts", label: "TTS 生成", icon: "🎤", color: "from-blue-500 to-cyan-500" },
    { id: "clone", label: "语音克隆", icon: "🎙️", color: "from-indigo-500 to-purple-500" },
    { id: "voices", label: "声音库", icon: "👥", color: "from-purple-500 to-pink-500" },
    { id: "training", label: "模型训练", icon: "✨", color: "from-orange-500 to-red-500" },
    { id: "batch", label: "批量配音", icon: "🎵", color: "from-green-500 to-teal-500" },
    { id: "settings", label: "设置", icon: "⚙️", color: "from-gray-500 to-gray-600" },
  ];

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100">
      {/* Sidebar */}
      <aside
        className={`flex flex-col border-r bg-white/80 backdrop-blur-sm transition-all ${
          sidebarOpen ? "w-64" : "w-16"
        }`}
      >
        {/* Logo */}
        <div className="flex items-center justify-between border-b p-4">
          {sidebarOpen && (
            <div className="flex items-center gap-2">
              <div className="rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 p-1.5">
                <span className="text-sm">🌊</span>
              </div>
              <div>
                <h1 className="text-sm font-bold">延边朝鲜语 TTS</h1>
              </div>
            </div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="rounded-lg p-1.5 hover:bg-gray-100"
          >
            {sidebarOpen ? "✕" : "☰"}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 p-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                activeTab === item.id
                  ? `bg-gradient-to-r ${item.color} text-white shadow-lg`
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              <span className="text-lg">{item.icon}</span>
              {sidebarOpen && <span>{item.label}</span>}
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t p-4">
          {sidebarOpen && (
            <div className="text-xs text-gray-500">
              <p>基于 Fish Speech</p>
              <p>Apache 2.0 License</p>
            </div>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {/* Header */}
        <header className="sticky top-0 z-10 border-b bg-white/80 backdrop-blur-sm px-6 py-4">
          <h2 className="text-xl font-bold">
            {navItems.find((i) => i.id === activeTab)?.label}
          </h2>
        </header>

        {/* Content Area */}
        <div className="p-6">
          {activeTab === "tts" && (
            <div className="mx-auto max-w-5xl">
              <div className="grid gap-6 lg:grid-cols-3">
                {/* Text Input */}
                <div className="lg:col-span-2 space-y-6">
                  <div className="rounded-2xl border bg-white p-6 shadow-lg">
                    <label className="mb-3 block font-semibold">输入文本</label>
                    <textarea
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                      placeholder="请输入要转换为语音的文本...支持中文、朝鲜语混合输入"
                      className="w-full min-h-[200px] rounded-xl border p-4 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                      disabled={generating}
                    />
                    <div className="mt-2 text-sm text-gray-500">
                      {text.length} 字符
                    </div>
                  </div>

                  {/* Audio Output */}
                  {audioUrl && (
                    <div className="rounded-2xl border bg-white p-6 shadow-lg">
                      <h3 className="mb-4 font-semibold">生成结果</h3>
                      <audio src={audioUrl} controls className="w-full" autoPlay />
                      <button
                        onClick={() => {
                          const a = document.createElement("a");
                          a.href = audioUrl;
                          a.download = `tts_${Date.now()}.wav`;
                          a.click();
                        }}
                        className="mt-4 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
                      >
                        📥 下载音频
                      </button>
                    </div>
                  )}
                </div>

                {/* Controls */}
                <div className="space-y-4">
                  <button
                    onClick={handleGenerate}
                    disabled={generating || !text.trim()}
                    className="w-full rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4 font-semibold text-white shadow-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {generating ? "⏳ 生成中..." : "▶️ 生成语音"}
                  </button>

                  {/* Voice Selection */}
                  <div className="rounded-2xl border bg-white p-5 shadow-lg">
                    <div className="mb-3 flex items-center justify-between">
                      <label className="font-semibold">🎙️ 选择声音</label>
                      {selectedVoiceForTTS && (
                        <button
                          onClick={() => setSelectedVoiceForTTS("")}
                          className="text-xs text-red-600 hover:text-red-800 hover:underline"
                        >
                          清除选择
                        </button>
                      )}
                    </div>
                    <select
                      value={selectedVoiceForTTS}
                      onChange={(e) => setSelectedVoiceForTTS(e.target.value)}
                      className="w-full rounded-lg border p-2.5 text-sm"
                      disabled={generating}
                    >
                      <option value="">使用默认声音</option>
                      {voices.filter(v => v.is_trained).map((voice) => (
                        <option key={voice.id} value={voice.id}>
                          {voice.name} ({voice.description || "无描述"})
                        </option>
                      ))}
                    </select>
                    {selectedVoiceForTTS && (
                      <div className="mt-2 text-xs text-gray-500">
                        ✅ 使用声音库中的 "{voices.find(v => v.id === selectedVoiceForTTS)?.name}"
                      </div>
                    )}
                  </div>

                  {/* Emotion */}
                  <div className="rounded-2xl border bg-white p-5 shadow-lg">
                    <label className="mb-3 block font-semibold">情感控制</label>
                    <select
                      value={emotion}
                      onChange={(e) => setEmotion(e.target.value)}
                      className="w-full rounded-lg border p-2.5 text-sm"
                      disabled={generating}
                    >
                      {emotions.map((e) => (
                        <option key={e.value} value={e.value}>{e.label}</option>
                      ))}
                    </select>
                  </div>

                  {/* Speed */}
                  <div className="rounded-2xl border bg-white p-5 shadow-lg">
                    <div className="mb-3 flex items-center justify-between">
                      <label className="font-semibold">语速</label>
                      <span className="text-sm font-medium text-blue-600">{speedFactor.toFixed(1)}x</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="2.0"
                      step="0.1"
                      value={speedFactor}
                      onChange={(e) => setSpeedFactor(parseFloat(e.target.value))}
                      className="w-full accent-blue-600"
                      disabled={generating}
                    />
                  </div>

                  {/* Pitch */}
                  <div className="rounded-2xl border bg-white p-5 shadow-lg">
                    <div className="mb-3 flex items-center justify-between">
                      <label className="font-semibold">音调</label>
                      <span className="text-sm font-medium text-purple-600">{pitchFactor.toFixed(2)}x</span>
                    </div>
                    <input
                      type="range"
                      min="0.8"
                      max="1.2"
                      step="0.05"
                      value={pitchFactor}
                      onChange={(e) => setPitchFactor(parseFloat(e.target.value))}
                      className="w-full accent-purple-600"
                      disabled={generating}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "clone" && (
            <div className="mx-auto max-w-4xl">
              <div className="rounded-2xl border bg-white p-8 shadow-lg">
                <div className="mb-6 flex items-center gap-3">
                  <span className="text-4xl">🎙️</span>
                  <div>
                    <h3 className="text-xl font-bold">创建语音克隆</h3>
                    <p className="text-sm text-gray-500">上传30分钟+的音频，训练专属声音模型</p>
                  </div>
                </div>

                <div className="space-y-6">
                  {/* Info */}
                  <div className="rounded-xl border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
                    <p className="font-medium">⚠️ 重要说明 - 语音克隆工作原理</p>
                    <ul className="mt-2 space-y-1 text-xs">
                      <li>• <strong>参考式语音克隆</strong>：系统使用您上传的音频作为"风格参考"来生成语音</li>
                      <li>• <strong>不是100%克隆</strong>：输出声音会是参考音频和基础模型的混合，约70-80%相似度</li>
                      <li>• <strong>关键要求</strong>：参考音频内容文本（"参考音频内容文本"）必须与音频实际说的话完全匹配</li>
                      <li>• <strong>音频质量</strong>：使用清晰、无背景噪音的高质量音频会获得更好效果</li>
                      <li>• <strong>时长建议</strong>：建议6-30秒的音频，会自动分割成6秒片段以增加参考点</li>
                    </ul>
                  </div>

                  {/* Voice Name */}
                  <div>
                    <label className="mb-2 block font-semibold">
                      声音名称 <span className="text-red-500">*</span>
                    </label>
                    <input
                      type="text"
                      value={voiceName}
                      onChange={(e) => setVoiceName(e.target.value)}
                      placeholder="例如: 延边女声_张明"
                      className="w-full rounded-lg border px-4 py-3 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                      disabled={uploading}
                    />
                  </div>

                  {/* Description */}
                  <div>
                    <label className="mb-2 block font-semibold">声音描述</label>
                    <textarea
                      value={voiceDescription}
                      onChange={(e) => setVoiceDescription(e.target.value)}
                      placeholder="描述这个声音的特点，例如：年轻女性，延边方言，语调柔和..."
                      className="w-full min-h-[80px] rounded-lg border px-4 py-3 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                      disabled={uploading}
                    />
                  </div>

                  {/* Reference Audio Content Text - NEW */}
                  <div>
                    <label className="mb-2 block font-semibold">
                      参考音频内容文本 <span className="text-red-500">*</span>
                    </label>
                    <div className="rounded-xl border-amber-200 bg-amber-50 p-3 text-sm text-amber-800 mb-2">
                      <p className="font-medium">⚠️ 重要提示</p>
                      <p className="mt-1 text-xs">请输入你上传的音频中<strong>实际说的话</strong>。这个文本必须与音频内容匹配，才能获得正确的克隆效果。</p>
                    </div>
                    <textarea
                      value={referenceText}
                      onChange={(e) => setReferenceText(e.target.value)}
                      placeholder="例如：你好，我是张明，很高兴认识大家。今天天气真好。"
                      className="w-full min-h-[80px] rounded-lg border px-4 py-3 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                      disabled={uploading}
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      💡 这段文本会被用来标记参考音频，确保它与音频内容一致非常重要。
                    </p>
                  </div>

                  {/* Audio File Upload */}
                  <div>
                    <label className="mb-2 block font-semibold">
                      音频文件 <span className="text-red-500">*</span>
                    </label>
                    <div className="rounded-lg border-2 border-dashed p-8 text-center transition-colors hover:border-indigo-400">
                      <input
                        type="file"
                        accept="audio/*"
                        onChange={handleFileSelect}
                        className="hidden"
                        id="audio-upload"
                        disabled={uploading}
                      />
                      <label htmlFor="audio-upload" className="cursor-pointer">
                        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-indigo-100">
                          <span className="text-2xl">📁</span>
                        </div>
                        <p className="font-medium text-gray-700">
                          {audioFile ? audioFile.name : "点击上传音频文件"}
                        </p>
                        <p className="mt-1 text-sm text-gray-500">
                          {audioFile
                            ? `${formatFileSize(audioFile.size)}`
                            : "支持 WAV/MP3/M4A 格式，最大 500MB"}
                        </p>
                        {audioFile && (
                          <div className="mt-2 inline-flex items-center gap-1 rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700">
                            <span>✓</span> 已选择
                          </div>
                        )}
                      </label>
                    </div>
                    <p className="mt-2 text-xs text-gray-500">
                      💡 推荐使用6-30秒的高质量音频。支持任意格式，会自动分割成6秒片段。
                    </p>

                    {/* Audio Preview Player - NEW */}
                    {audioPreviewUrl && (
                      <div className="mt-4 rounded-xl border border-indigo-200 bg-indigo-50 p-4">
                        <div className="mb-2 flex items-center gap-2">
                          <span className="text-lg">🎧</span>
                          <span className="font-medium text-indigo-900">音频预览</span>
                        </div>
                        <audio
                          src={audioPreviewUrl}
                          controls
                          className="w-full"
                        />
                        <p className="mt-2 text-xs text-indigo-700">
                          💡 请试听一下，确认这是你想要克隆的声音
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Training Parameters */}
                  <div className="rounded-xl border bg-gray-50 p-5">
                    <h4 className="mb-4 font-semibold">训练参数</h4>
                    <div className="grid gap-4 sm:grid-cols-3">
                      {/* Max Steps */}
                      <div>
                        <label className="mb-2 block text-sm font-medium">
                          训练步数
                        </label>
                        <input
                          type="number"
                          min="1000"
                          max="20000"
                          step="1000"
                          value={trainingMaxSteps}
                          onChange={(e) => setTrainingMaxSteps(parseInt(e.target.value) || 5000)}
                          className="w-full rounded-lg border px-3 py-2 text-sm"
                          disabled={uploading}
                        />
                        <p className="mt-1 text-xs text-gray-500">推荐: 5000-10000</p>
                      </div>

                      {/* Learning Rate */}
                      <div>
                        <label className="mb-2 block text-sm font-medium">
                          学习率
                        </label>
                        <select
                          value={trainingLearningRate}
                          onChange={(e) => setTrainingLearningRate(parseFloat(e.target.value))}
                          className="w-full rounded-lg border px-3 py-2 text-sm"
                          disabled={uploading}
                        >
                          <option value="1e-5">0.00001 (慢)</option>
                          <option value="5e-5">0.00005</option>
                          <option value="1e-4">0.0001 (推荐)</option>
                          <option value="2e-4">0.0002</option>
                          <option value="5e-4">0.0005 (快)</option>
                        </select>
                      </div>

                      {/* Batch Size */}
                      <div>
                        <label className="mb-2 block text-sm font-medium">
                          批量大小
                        </label>
                        <select
                          value={trainingBatchSize}
                          onChange={(e) => setTrainingBatchSize(parseInt(e.target.value))}
                          className="w-full rounded-lg border px-3 py-2 text-sm"
                          disabled={uploading}
                        >
                          <option value="8">8</option>
                          <option value="16">16 (推荐)</option>
                          <option value="32">32</option>
                          <option value="64">64</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Upload Progress */}
                  {uploading && (
                    <div className="rounded-xl border bg-indigo-50 p-5">
                      <div className="mb-2 flex items-center justify-between">
                        <span className="font-medium text-indigo-900">
                          {uploadProgress < 100 ? "正在上传..." : "正在创建训练任务..."}
                        </span>
                        <span className="text-sm font-semibold text-indigo-600">
                          {uploadProgress.toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-3 overflow-hidden rounded-full bg-indigo-200">
                        <div
                          className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-300"
                          style={{ width: `${uploadProgress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Submit Button */}
                  <button
                    onClick={handleVoiceClone}
                    disabled={uploading || !voiceName.trim() || !audioFile}
                    className="w-full rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4 font-semibold text-white shadow-lg hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {uploading ? "⏳ 上传中..." : "🚀 开始训练"}
                  </button>

                  {/* Info */}
                  <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
                    <p className="font-medium">⚡ 训练提示</p>
                    <ul className="mt-2 space-y-1 text-xs">
                      <li>• 训练时间约2-4小时（取决于硬件和数据量）</li>
                      <li>• 推荐使用 RTX 3060 或更高显卡</li>
                      <li>• 训练完成后可在「声音库」中查看</li>
                      <li>• 训练进度可在「模型训练」页面查看</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "voices" && (
            <div className="mx-auto max-w-6xl">
              <div className="rounded-2xl border bg-white p-6 shadow-lg">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="font-semibold">声音列表 ({voices.length})</h3>
                  <button
                    onClick={loadData}
                    className="rounded-lg px-3 py-1.5 text-sm font-medium text-blue-600 hover:bg-blue-50"
                  >
                    🔄 刷新
                  </button>
                </div>
                {voices.length === 0 ? (
                  <div className="py-8 text-center text-gray-500">
                    还没有声音，请先在「语音克隆」页面创建
                  </div>
                ) : (
                  <div>
                    {/* Test Controls Panel */}
                    <div className="mb-6 rounded-xl border bg-gray-50 p-4">
                      <h4 className="mb-3 font-semibold text-sm">🎛️ 测试参数控制</h4>
                      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                        {/* Test Text */}
                        <div className="sm:col-span-2">
                          <label className="mb-1 block text-xs font-medium">测试文本</label>
                          <input
                            type="text"
                            value={voiceTestText}
                            onChange={(e) => setVoiceTestText(e.target.value)}
                            placeholder="输入要测试的文本..."
                            className="w-full rounded-lg border px-3 py-2 text-sm"
                          />
                        </div>
                        {/* Emotion */}
                        <div>
                          <label className="mb-1 block text-xs font-medium">情感</label>
                          <select
                            value={voiceTestEmotion}
                            onChange={(e) => setVoiceTestEmotion(e.target.value)}
                            className="w-full rounded-lg border px-3 py-2 text-sm"
                          >
                            <option value="">默认</option>
                            <option value="happy">快乐</option>
                            <option value="sad">悲伤</option>
                            <option value="angry">愤怒</option>
                            <option value="whisper">耳语</option>
                            <option value="shout">呼喊</option>
                          </select>
                        </div>
                        {/* Speed & Pitch */}
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="mb-1 block text-xs font-medium">语速</label>
                            <input
                              type="number"
                              min="0.5"
                              max="2.0"
                              step="0.1"
                              value={voiceTestSpeed}
                              onChange={(e) => setVoiceTestSpeed(parseFloat(e.target.value) || 1.0)}
                              className="w-full rounded-lg border px-2 py-2 text-sm"
                            />
                          </div>
                          <div>
                            <label className="mb-1 block text-xs font-medium">音调</label>
                            <input
                              type="number"
                              min="0.8"
                              max="1.2"
                              step="0.05"
                              value={voiceTestPitch}
                              onChange={(e) => setVoiceTestPitch(parseFloat(e.target.value) || 1.0)}
                              className="w-full rounded-lg border px-2 py-2 text-sm"
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Voice Grid */}
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                      {voices.map((voice) => {
                        const audioCount = voice.audio_files?.length || 1;
                        return (
                          <div key={voice.id} className={`rounded-xl border p-4 hover:shadow-md transition-shadow ${
                            selectedVoiceForTest?.id === voice.id ? "ring-2 ring-indigo-500" : ""
                          }`}>
                            <div className="mb-2 flex items-start justify-between">
                              <div className="flex-1">
                                <h4 className="font-semibold">{voice.name}</h4>
                                <p className="text-sm text-gray-500">{voice.description || "无描述"}</p>
                              </div>
                              <div className="flex items-center gap-2">
                                <span className={`rounded-full px-2 py-1 text-xs font-medium ${
                                  voice.is_trained ? "bg-green-100 text-green-700" : "bg-yellow-100 text-yellow-700"
                                }`}>
                                  {voice.is_trained ? "✓ 已训练" : "训练中"}
                                </span>
                                <button
                                  onClick={() => handleDeleteVoice(voice.id, voice.name)}
                                  className="rounded p-1 text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors"
                                  title="删除声音"
                                >
                                  🗑️
                                </button>
                              </div>
                            </div>
                            <div className="mt-3 text-xs text-gray-400 space-y-1">
                              <p>ID: {voice.id}</p>
                              <p>语言: {voice.language || "延边朝鲜语"}</p>
                              <p>参考音频: {audioCount} 个片段</p>
                              {voice.duration && <p>时长: {voice.duration.toFixed(1)}秒</p>}
                              {voice.created_at && <p>创建: {new Date(voice.created_at).toLocaleDateString()}</p>}
                            </div>
                            <div className="mt-4 space-y-2">
                              <button
                                onClick={() => handleTestVoice(voice.id, voice)}
                                disabled={testingVoice === voice.id || !voice.is_trained}
                                className={`w-full rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                                  testingVoice === voice.id
                                    ? "bg-gray-100 text-gray-400 cursor-wait"
                                    : voice.is_trained
                                    ? "bg-indigo-600 text-white hover:bg-indigo-700"
                                    : "bg-gray-100 text-gray-400 cursor-not-allowed"
                                }`}
                              >
                                {testingVoice === voice.id ? "⏳ 生成中..." : "▶️ 测试播放"}
                              </button>

                              {/* Always show audio player for selected voice, with loading state */}
                              {selectedVoiceForTest?.id === voice.id && (
                                <>
                                  {testingVoice === voice.id ? (
                                    <div className="w-full rounded-lg bg-gray-100 p-3 text-center">
                                      <div className="inline-block animate-spin mr-2">⏳</div>
                                      <span className="text-sm text-gray-600">正在生成音频...</span>
                                    </div>
                                  ) : testAudioUrl ? (
                                    <div className="rounded-lg border bg-gray-50 p-2">
                                      <audio
                                        src={testAudioUrl}
                                        controls
                                        className="w-full"
                                        data-voice-id={voice.id}
                                      />
                                    </div>
                                  ) : null}
                                </>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "training" && (
            <div className="mx-auto max-w-6xl">
              <div className="rounded-2xl border bg-white p-6 shadow-lg">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="font-semibold">训练任务 ({trainingTasks.length})</h3>
                  <button
                    onClick={loadData}
                    className="rounded-lg px-3 py-1.5 text-sm font-medium text-blue-600 hover:bg-blue-50"
                  >
                    🔄 刷新
                  </button>
                </div>
                {trainingTasks.length === 0 ? (
                  <div className="py-8 text-center text-gray-500">
                    还没有训练任务，请先在「语音克隆」页面创建
                  </div>
                ) : (
                  <div className="space-y-4">
                    {trainingTasks.map((task) => {
                      // Calculate progress percentage
                      const progressPercent = task.progress ? Math.round(task.progress * 100) : Math.round(((task.current_step || 0) / (task.total_steps || 1)) * 100);
                      const isTraining = task.status === "running" || task.status === "training" || task.status === "preparing_data" || task.status === "extracting_vq" || task.status === "building_dataset" || task.status === "merging_weights";
                      return (
                        <div key={task.task_id} className="rounded-xl border p-4">
                          <div className="mb-3 flex items-center justify-between">
                            <div>
                              <h4 className="font-semibold">{task.voice_name || task.voice_id}</h4>
                              <p className="text-xs text-gray-500">任务ID: {task.task_id}</p>
                            </div>
                            <span className={`rounded-full px-3 py-1.5 text-sm font-medium ${
                              isTraining ? "bg-blue-100 text-blue-700" :
                              task.status === "completed" ? "bg-green-100 text-green-700" :
                              task.status === "failed" || task.status === "cancelled" ? "bg-red-100 text-red-700" :
                              "bg-gray-100 text-gray-700"
                            }`}>
                              {isTraining ? "🔄 训练中" :
                               task.status === "completed" ? "✅ 已完成" :
                               task.status === "failed" ? "❌ 失败" :
                               task.status === "cancelled" ? "⏹️ 已取消" : task.status}
                            </span>
                          </div>

                          {/* Training stages info */}
                          {isTraining && (
                            <div className="mb-3 rounded-lg bg-blue-50 p-3 text-sm">
                              <div className="font-medium text-blue-900">
                                🔄 LoRA 训练进行中
                              </div>
                              <div className="mt-1 text-xs text-blue-700">
                                预计需要 30-60 分钟，训练完成后语音相似度将达到 90-95%
                              </div>
                            </div>
                          )}

                          {/* Progress bar */}
                          {(isTraining || task.status === "completed") && (
                            <div>
                              <div className="mb-2 h-3 overflow-hidden rounded-full bg-gray-200">
                                <div
                                  className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 transition-all"
                                  style={{ width: `${progressPercent}%` }}
                                />
                              </div>
                              <div className="flex items-center justify-between text-sm">
                                <span className="text-gray-600">
                                  {task.progress ? `${progressPercent}%` : `${task.current_step || 0} / ${task.total_steps || 0} 步`}
                                </span>
                                <span className="font-medium text-blue-600">
                                  {progressPercent}%
                                </span>
                              </div>
                              {/* Current step info */}
                              {task.current_step && (
                                <div className="mt-2 text-xs text-gray-600">
                                  当前步骤: {task.current_step}
                                </div>
                              )}
                            </div>
                          )}

                          {task.status === "completed" && (
                            <div className="mt-3 rounded-lg bg-green-50 p-3 text-sm text-green-700">
                              ✅ LoRA 训练完成! 声音已添加到声音库，相似度 ~90-95%
                            </div>
                          )}
                          {task.status === "failed" && (
                            <div className="mt-3 rounded-lg bg-red-50 p-3 text-sm text-red-700">
                              ❌ 训练失败: {task.error || "未知错误"}
                            </div>
                          )}
                          {task.status === "cancelled" && (
                            <div className="mt-3 rounded-lg bg-gray-50 p-3 text-sm text-gray-700">
                              ⏹️ 训练已取消
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "batch" && (
            <div className="mx-auto max-w-6xl">
              <div className="rounded-2xl border bg-white p-6 shadow-lg">
                <h3 className="mb-4 font-semibold">批量配音任务 ({batchJobs.length})</h3>
                {batchJobs.length === 0 ? (
                  <div className="py-8 text-center text-gray-500">
                    还没有批量任务
                  </div>
                ) : (
                  <div className="space-y-4">
                    {batchJobs.map((job) => (
                      <div key={job.job_id} className="rounded-xl border p-4">
                        <div className="mb-2 flex items-center gap-2">
                          <h4 className="font-semibold">{job.name}</h4>
                          <span className="text-sm">
                            {job.status === "running" ? "🔄 进行中" : job.status === "completed" ? "✅ 已完成" : job.status}
                          </span>
                        </div>
                        {job.status === "running" && job.total_items > 0 && (
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-green-600 h-2 rounded-full"
                              style={{ width: `${(job.completed_items / job.total_items) * 100}%` }}
                            />
                          </div>
                        )}
                        <p className="text-xs text-gray-400 mt-2">
                          {job.completed_items} / {job.total_items} 项
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "settings" && (
            <div className="mx-auto max-w-2xl space-y-6">
              <div className="rounded-2xl border bg-white p-6 shadow-lg">
                <h3 className="mb-4 font-semibold">API 配置</h3>
                <div className="space-y-4">
                  <div>
                    <label className="mb-2 block text-sm font-medium">API 地址</label>
                    <input
                      value="http://localhost:7860"
                      readOnly
                      className="w-full rounded-lg border bg-gray-50 px-3 py-2 text-sm"
                    />
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border bg-white p-6 shadow-lg">
                <h3 className="mb-4 font-semibold">关于</h3>
                <div className="space-y-2 text-sm text-gray-600">
                  <p>🌊 延边朝鲜语语音克隆与TTS系统</p>
                  <p>基于 Fish Speech 框架构建</p>
                  <p>许可证: Apache 2.0</p>
                  <div className="pt-2 flex gap-4">
                    <a href="https://github.com/fishaudio/fish-speech" target="_blank" className="text-blue-600 hover:underline">
                      GitHub
                    </a>
                    <a href="https://fish.audio" target="_blank" className="text-blue-600 hover:underline">
                      Fish Audio
                    </a>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
