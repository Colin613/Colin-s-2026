"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Upload,
  Play,
  Trash2,
  Plus,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Sparkles,
} from "lucide-react";
import { voicesApi, trainingApi, referencesApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatDuration, formatDate } from "@/lib/utils";

export default function VoiceStudioPage() {
  const [voices, setVoices] = useState<any[]>([]);
  const [trainingTasks, setTrainingTasks] = useState<any[]>([]);
  const [references, setReferences] = useState<string[]>([]);

  // New voice form
  const [newVoiceId, setNewVoiceId] = useState("");
  const [newVoiceName, setNewVoiceName] = useState("");
  const [newVoiceDesc, setNewVoiceDesc] = useState("");

  // Reference upload
  const [refId, setRefId] = useState("");
  const [refAudio, setRefAudio] = useState<File | null>(null);
  const [refText, setRefText] = useState("");
  const [uploading, setUploading] = useState(false);

  // Training form
  const [trainVoiceId, setTrainVoiceId] = useState("");
  const [trainDataPath, setTrainDataPath] = useState("");
  const [trainMaxSteps, setTrainMaxSteps] = useState(5000);
  const [trainBatchSize, setTrainBatchSize] = useState(16);

  const [loading, setLoading] = useState(true);

  // Load data
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [voicesData, tasksData, refsData] = await Promise.all([
        voicesApi.list(),
        trainingApi.list(),
        referencesApi.list(),
      ]);
      setVoices(voicesData);
      setTrainingTasks(tasksData);
      setReferences(refsData);
    } catch (error) {
      console.error("Failed to load data:", error);
    } finally {
      setLoading(false);
    }
  };

  // Create new voice
  const handleCreateVoice = async () => {
    if (!newVoiceId || !newVoiceName) return;

    try {
      await voicesApi.create({
        id: newVoiceId,
        name: newVoiceName,
        description: newVoiceDesc,
        language: "ko",
      });
      setNewVoiceId("");
      setNewVoiceName("");
      setNewVoiceDesc("");
      loadData();
    } catch (error) {
      alert(`创建失败: ${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  // Delete voice
  const handleDeleteVoice = async (voiceId: string) => {
    if (!confirm(`确定要删除声音 "${voiceId}" 吗？`)) return;

    try {
      await voicesApi.delete(voiceId);
      loadData();
    } catch (error) {
      alert(`删除失败: ${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  // Upload reference audio
  const handleUploadReference = async () => {
    if (!refId || !refAudio || !refText) return;

    setUploading(true);
    try {
      await referencesApi.add(refId, refAudio, refText);
      setRefId("");
      setRefAudio(null);
      setRefText("");
      loadData();
    } catch (error) {
      alert(`上传失败: ${error instanceof Error ? error.message : "未知错误"}`);
    } finally {
      setUploading(false);
    }
  };

  // Delete reference
  const handleDeleteReference = async (refId: string) => {
    try {
      await referencesApi.delete(refId);
      loadData();
    } catch (error) {
      alert(`删除失败: ${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  // Start training
  const handleStartTraining = async () => {
    if (!trainVoiceId || !trainDataPath) return;

    try {
      const result = await trainingApi.start({
        voice_id: trainVoiceId,
        data_path: trainDataPath,
        max_steps: trainMaxSteps,
        batch_size: trainBatchSize,
      });
      alert(`训练任务已创建: ${result.task_id}`);
      loadData();
    } catch (error) {
      alert(`启动失败: ${error instanceof Error ? error.message : "未知错误"}`);
    }
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />;
      case "running":
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  // Get status text
  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      pending: "等待中",
      running: "训练中",
      completed: "已完成",
      failed: "失败",
      cancelled: "已取消",
    };
    return statusMap[status] || status;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
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
        <div className="mx-auto max-w-6xl">
          {/* Page Header */}
          <div className="mb-8">
            <div className="flex items-center gap-3">
              <div className="rounded-full bg-gradient-to-br from-purple-500 to-pink-600 p-2">
                <Sparkles className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">语音克隆工作室</h1>
                <p className="text-muted-foreground">
                  创建、训练和管理自定义声音模型
                </p>
              </div>
            </div>
          </div>

          <Tabs defaultValue="voices" className="space-y-6">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="voices">声音库</TabsTrigger>
              <TabsTrigger value="references">参考音频</TabsTrigger>
              <TabsTrigger value="training">模型训练</TabsTrigger>
              <TabsTrigger value="tasks">训练任务</TabsTrigger>
            </TabsList>

            {/* Voices Tab */}
            <TabsContent value="voices" className="space-y-6">
              {/* Create New Voice */}
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <h3 className="mb-4 font-semibold">创建新声音</h3>
                <div className="grid gap-4 md:grid-cols-4">
                  <div>
                    <Label>声音 ID</Label>
                    <Input
                      placeholder="my_voice"
                      value={newVoiceId}
                      onChange={(e) => setNewVoiceId(e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>声音名称</Label>
                    <Input
                      placeholder="我的声音"
                      value={newVoiceName}
                      onChange={(e) => setNewVoiceName(e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>描述</Label>
                    <Input
                      placeholder="可选"
                      value={newVoiceDesc}
                      onChange={(e) => setNewVoiceDesc(e.target.value)}
                    />
                  </div>
                  <div className="flex items-end">
                    <Button onClick={handleCreateVoice} className="w-full">
                      <Plus className="mr-2 h-4 w-4" />
                      创建
                    </Button>
                  </div>
                </div>
              </div>

              {/* Voices List */}
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <h3 className="mb-4 font-semibold">声音列表 ({voices.length})</h3>
                {loading ? (
                  <div className="flex justify-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : voices.length === 0 ? (
                  <div className="py-8 text-center text-muted-foreground">
                    还没有声音，创建一个吧！
                  </div>
                ) : (
                  <div className="space-y-3">
                    {voices.map((voice) => (
                      <div
                        key={voice.id}
                        className="flex items-center justify-between rounded-lg border p-4"
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <h4 className="font-semibold">{voice.name}</h4>
                            {voice.is_trained && (
                              <span className="rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700 dark:bg-green-900/30 dark:text-green-400">
                                已训练
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground">{voice.description}</p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            ID: {voice.id} | 语言: {voice.language} | 创建: {formatDate(voice.created_at)}
                          </p>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteVoice(voice.id)}
                        >
                          <Trash2 className="h-4 w-4 text-red-500" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </TabsContent>

            {/* References Tab */}
            <TabsContent value="references" className="space-y-6">
              {/* Upload Reference */}
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <h3 className="mb-4 font-semibold">上传参考音频</h3>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <Label>参考 ID</Label>
                    <Input
                      placeholder="reference_001"
                      value={refId}
                      onChange={(e) => setRefId(e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>音频文件</Label>
                    <Input
                      type="file"
                      accept="audio/*"
                      onChange={(e) => setRefAudio(e.target.files?.[0] || null)}
                    />
                  </div>
                  <div className="md:col-span-2">
                    <Label>参考文本</Label>
                    <Input
                      placeholder="这是参考音频对应的文本内容..."
                      value={refText}
                      onChange={(e) => setRefText(e.target.value)}
                    />
                  </div>
                </div>
                <div className="mt-4">
                  <Button onClick={handleUploadReference} disabled={uploading || !refId || !refAudio || !refText}>
                    {uploading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        上传中...
                      </>
                    ) : (
                      <>
                        <Upload className="mr-2 h-4 w-4" />
                        上传
                      </>
                    )}
                  </Button>
                </div>
              </div>

              {/* References List */}
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <h3 className="mb-4 font-semibold">参考音频列表 ({references.length})</h3>
                {references.length === 0 ? (
                  <div className="py-8 text-center text-muted-foreground">
                    还没有参考音频
                  </div>
                ) : (
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    {references.map((ref) => (
                      <div
                        key={ref}
                        className="flex items-center justify-between rounded-lg border p-4"
                      >
                        <div className="flex items-center gap-3">
                          <div className="rounded-full bg-purple-100 p-2 dark:bg-purple-900/30">
                            <Play className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                          </div>
                          <span className="font-medium">{ref}</span>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteReference(ref)}
                        >
                          <Trash2 className="h-4 w-4 text-red-500" />
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </TabsContent>

            {/* Training Tab */}
            <TabsContent value="training" className="space-y-6">
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <h3 className="mb-4 font-semibold">启动 LoRA 微调训练</h3>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <Label>声音 ID</Label>
                    <Input
                      placeholder="my_voice"
                      value={trainVoiceId}
                      onChange={(e) => setTrainVoiceId(e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>训练数据路径</Label>
                    <Input
                      placeholder="data/yanbian_voice"
                      value={trainDataPath}
                      onChange={(e) => setTrainDataPath(e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>最大训练步数</Label>
                    <Input
                      type="number"
                      value={trainMaxSteps}
                      onChange={(e) => setTrainMaxSteps(parseInt(e.target.value) || 5000)}
                    />
                  </div>
                  <div>
                    <Label>批次大小</Label>
                    <Input
                      type="number"
                      value={trainBatchSize}
                      onChange={(e) => setTrainBatchSize(parseInt(e.target.value) || 16)}
                    />
                  </div>
                </div>
                <div className="mt-4">
                  <Button onClick={handleStartTraining} disabled={!trainVoiceId || !trainDataPath}>
                    <Sparkles className="mr-2 h-4 w-4" />
                    开始训练
                  </Button>
                </div>
                <div className="mt-4 rounded-lg bg-blue-50 p-4 text-sm dark:bg-blue-900/20">
                  <p className="font-medium text-blue-900 dark:text-blue-100">训练说明：</p>
                  <ul className="mt-2 space-y-1 text-blue-800 dark:text-blue-200">
                    <li>• 数据路径应包含按说话人组织的音频文件和对应的 .lab 转写文件</li>
                    <li>• 建议使用 30 分钟以上的高质量音频进行训练</li>
                    <li>• 训练时间约 2-4 小时（RTX 3060）</li>
                    <li>• 可使用 <code className="rounded bg-blue-100 px-1 py-0.5 dark:bg-blue-800">python tools/yanbian_finetune.py</code> 命令进行完整训练</li>
                  </ul>
                </div>
              </div>
            </TabsContent>

            {/* Tasks Tab */}
            <TabsContent value="tasks" className="space-y-6">
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="font-semibold">训练任务 ({trainingTasks.length})</h3>
                  <Button variant="outline" size="sm" onClick={loadData}>
                    刷新
                  </Button>
                </div>
                {trainingTasks.length === 0 ? (
                  <div className="py-8 text-center text-muted-foreground">
                    还没有训练任务
                  </div>
                ) : (
                  <div className="space-y-4">
                    {trainingTasks.map((task) => (
                      <div
                        key={task.task_id}
                        className="rounded-lg border p-4"
                      >
                        <div className="mb-3 flex items-start justify-between">
                          <div>
                            <div className="flex items-center gap-2">
                              <h4 className="font-semibold">{task.voice_id}</h4>
                              {getStatusIcon(task.status)}
                              <span className="text-sm text-muted-foreground">
                                {getStatusText(task.status)}
                              </span>
                            </div>
                            <p className="mt-1 text-xs text-muted-foreground">
                              任务 ID: {task.task_id}
                            </p>
                          </div>
                          {task.status === "running" && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={async () => {
                                await trainingApi.cancel(task.task_id);
                                loadData();
                              }}
                            >
                              取消
                            </Button>
                          )}
                        </div>

                        {/* Progress Bar */}
                        {task.status === "running" && (
                          <div className="space-y-2">
                            <Progress value={(task.progress || 0) * 100} />
                            <div className="flex justify-between text-xs text-muted-foreground">
                              <span>进度: {task.progress ? Math.round(task.progress * 100) : 0}%</span>
                              <span>
                                步数: {task.current_step} / {task.total_steps}
                              </span>
                            </div>
                          </div>
                        )}

                        {/* Error Message */}
                        {task.status === "failed" && task.error_message && (
                          <div className="mt-2 rounded-lg bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/20 dark:text-red-400">
                            {task.error_message}
                          </div>
                        )}

                        {/* Timestamps */}
                        <div className="mt-3 flex flex-wrap gap-4 text-xs text-muted-foreground">
                          <span>创建: {formatDate(task.created_at)}</span>
                          {task.started_at && <span>开始: {formatDate(task.started_at)}</span>}
                          {task.completed_at && <span>完成: {formatDate(task.completed_at)}</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
