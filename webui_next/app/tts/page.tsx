"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowLeft, Mic, Play, Download, Loader2 } from "lucide-react";
import { ttsApi } from "@/lib/api";
import { downloadBlob } from "@/lib/utils";

export default function TTSPage() {
  const [text, setText] = useState("");
  const [referenceId, setReferenceId] = useState("");
  const [speedFactor, setSpeedFactor] = useState(1.0);
  const [pitchFactor, setPitchFactor] = useState(1.0);
  const [emotionIntensity, setEmotionIntensity] = useState(1.0);
  const [volumeGain, setVolumeGain] = useState(1.0);
  const [emotion, setEmotion] = useState("");

  const [generating, setGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const emotions = [
    { value: "", label: "默认" },
    { value: "(angry)", label: "愤怒" },
    { value: "(sad)", label: "悲伤" },
    { value: "(happy)", label: "快乐" },
    { value: "(excited)", label: "兴奋" },
    { value: "(surprised)", label: "惊讶" },
    { value: "(nervous)", label: "紧张" },
    { value: "(confident)", label: "自信" },
  ];

  const handleGenerate = async () => {
    if (!text.trim()) {
      return;
    }

    setGenerating(true);
    setAudioUrl(null);

    try {
      const textWithEmotion = emotion ? `${emotion} ${text}` : text;

      const blob = await ttsApi.generate({
        text: textWithEmotion,
        reference_id: referenceId || null,
        speed_factor: speedFactor,
        pitch_factor: pitchFactor,
        emotion_intensity: emotionIntensity,
        volume_gain: volumeGain,
        format: "wav",
      });

      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
    } catch (error) {
      console.error("TTS generation failed:", error);
      alert(`生成失败: ${error instanceof Error ? error.message : "未知错误"}`);
    } finally {
      setGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!audioUrl) return;

    const a = document.createElement("a");
    a.href = audioUrl;
    a.download = `yanbian_tts_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {/* Header */}
      <header className="border-b bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            返回首页
          </Link>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="mx-auto max-w-4xl">
          <div className="mb-8">
            <h1 className="mb-2 text-3xl font-bold">TTS 语音生成</h1>
            <p className="text-muted-foreground">
              输入文本，调整参数，生成高质量的延边朝鲜语语音
            </p>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            {/* Main Input Area */}
            <div className="lg:col-span-2 space-y-6">
              {/* Text Input */}
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <label className="mb-2 block font-semibold">
                  输入文本
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="请输入要转换为语音的文本..."
                  className="w-full min-h-[200px] rounded-lg border bg-transparent p-4 text-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                  disabled={generating}
                />
                <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                  <span>{text.length} 字符</span>
                  <span>支持中文、朝鲜语混合输入</span>
                </div>
              </div>

              {/* Audio Output */}
              {audioUrl && (
                <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="font-semibold">生成结果</h3>
                    <button
                      onClick={handleDownload}
                      className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
                    >
                      <Download className="h-4 w-4" />
                      下载音频
                    </button>
                  </div>
                  <audio
                    src={audioUrl}
                    controls
                    className="w-full"
                    autoPlay
                  />
                </div>
              )}
            </div>

            {/* Controls Sidebar */}
            <div className="space-y-6">
              {/* Generate Button */}
              <button
                onClick={handleGenerate}
                disabled={generating || !text.trim()}
                className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4 font-semibold text-white shadow-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {generating ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    生成中...
                  </>
                ) : (
                  <>
                    <Play className="h-5 w-5" />
                    生成语音
                  </>
                )}
              </button>

              {/* Emotion Selection */}
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <label className="mb-3 block font-semibold">
                  情感控制
                </label>
                <select
                  value={emotion}
                  onChange={(e) => setEmotion(e.target.value)}
                  className="w-full rounded-lg border bg-transparent p-3 text-sm focus:border-blue-500 focus:outline-none"
                  disabled={generating}
                >
                  {emotions.map((e) => (
                    <option key={e.value} value={e.value}>
                      {e.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Speed Control */}
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <label className="mb-3 block font-semibold">
                  语速调节
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={speedFactor}
                  onChange={(e) => setSpeedFactor(parseFloat(e.target.value))}
                  className="w-full"
                  disabled={generating}
                />
                <div className="mt-2 text-center text-sm text-muted-foreground">
                  {speedFactor.toFixed(1)}x
                </div>
              </div>

              {/* Pitch Control */}
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <label className="mb-3 block font-semibold">
                  音调调节
                </label>
                <input
                  type="range"
                  min="0.8"
                  max="1.2"
                  step="0.05"
                  value={pitchFactor}
                  onChange={(e) => setPitchFactor(parseFloat(e.target.value))}
                  className="w-full"
                  disabled={generating}
                />
                <div className="mt-2 text-center text-sm text-muted-foreground">
                  {pitchFactor.toFixed(2)}x
                </div>
              </div>

              {/* Emotion Intensity */}
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <label className="mb-3 block font-semibold">
                  情感强度
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="1.5"
                  step="0.1"
                  value={emotionIntensity}
                  onChange={(e) => setEmotionIntensity(parseFloat(e.target.value))}
                  className="w-full"
                  disabled={generating}
                />
                <div className="mt-2 text-center text-sm text-muted-foreground">
                  {emotionIntensity.toFixed(1)}x
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
