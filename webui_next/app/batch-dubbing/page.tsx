"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Upload,
  Play,
  Plus,
  Trash2,
  Loader2,
  FileAudio,
  CheckCircle2,
  XCircle,
  Clock,
  Download,
  Users,
  Settings,
} from "lucide-react";
import { batchApi, voicesApi } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatDate, formatProgress, formatDuration } from "@/lib/utils";

interface VoiceMapping {
  character: string;
  voiceId: string;
}

export default function BatchDubbingPage() {
  const [voices, setVoices] = useState<any[]>([]);
  const [batchJobs, setBatchJobs] = useState<any[]>([]);

  // New job form
  const [jobName, setJobName] = useState("");
  const [subtitleFile, setSubtitleFile] = useState<File | null>(null);
  const [voiceMappings, setVoiceMappings] = useState<VoiceMapping[]>([]);
  const [newCharacter, setNewCharacter] = useState("");
  const [newVoiceId, setNewVoiceId] = useState("");

  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  // Load data
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [voicesData, jobsData] = await Promise.all([
        voicesApi.list(),
        batchApi.list(),
      ]);
      setVoices(voicesData);
      setBatchJobs(jobsData);
    } catch (error) {
      console.error("Failed to load data:", error);
    } finally {
      setLoading(false);
    }
  };

  // Add voice mapping
  const handleAddMapping = () => {
    if (!newCharacter || !newVoiceId) return;

    setVoiceMappings([...voiceMappings, { character: newCharacter, voiceId: newVoiceId }]);
    setNewCharacter("");
    setNewVoiceId("");
  };

  // Remove voice mapping
  const handleRemoveMapping = (index: number) => {
    setVoiceMappings(voiceMappings.filter((_, i) => i !== index));
  };

  // Create batch job
  const handleCreateJob = async () => {
    if (!jobName || !subtitleFile || voiceMappings.length === 0) {
      alert("请填写所有必填项");
      return;
    }

    setCreating(true);
    try {
      // Create a mapping object from the array
      const mappingObj: Record<string, string> = {};
      voiceMappings.forEach((m) => {
        mappingObj[m.character] = m.voiceId;
      });

      // For demo, we'll use a fake path since we can't upload files to the server
      const result = await batchApi.create({
        name: jobName,
        subtitle_file: subtitleFile.name,
        voice_mappings: mappingObj,
        output_format: "wav",
      });

      alert(`批量任务已创建: ${result.job_id}\n注意：实际文件上传需要后端支持`);

      setJobName("");
      setSubtitleFile(null);
      setVoiceMappings([]);
      loadData();
    } catch (error) {
      alert(`创建失败: ${error instanceof Error ? error.message : "未知错误"}`);
    } finally {
      setCreating(false);
    }
  };

  // Delete job
  const handleDeleteJob = async (jobId: string) => {
    if (!confirm("确定要删除此任务吗？")) return;

    try {
      await batchApi.cancel(jobId);
      loadData();
    } catch (error) {
      alert(`删除失败: ${error instanceof Error ? error.message : "未知错误"}`);
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
      case "partial":
        return <CheckCircle2 className="h-4 w-4 text-yellow-500" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  // Get status text
  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      pending: "等待中",
      running: "处理中",
      completed: "已完成",
      failed: "失败",
      cancelled: "已取消",
      partial: "部分完成",
    };
    return statusMap[status] || status;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
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
              <div className="rounded-full bg-gradient-to-br from-green-500 to-teal-600 p-2">
                <FileAudio className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">批量配音工作台</h1>
                <p className="text-muted-foreground">
                  上传 SRT 字幕文件，为不同角色分配声音，批量生成配音
                </p>
              </div>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            {/* Create Job Form */}
            <div className="lg:col-span-2 space-y-6">
              <Tabs defaultValue="create" className="space-y-6">
                <TabsList>
                  <TabsTrigger value="create">创建任务</TabsTrigger>
                  <TabsTrigger value="help">使用说明</TabsTrigger>
                </TabsList>

                <TabsContent value="create" className="space-y-6">
                  {/* Job Info */}
                  <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                    <h3 className="mb-4 font-semibold">任务信息</h3>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div>
                        <Label>任务名称 *</Label>
                        <Input
                          placeholder="例如：电视剧第1集配音"
                          value={jobName}
                          onChange={(e) => setJobName(e.target.value)}
                        />
                      </div>
                      <div>
                        <Label>SRT 字幕文件 *</Label>
                        <div className="flex items-center gap-2">
                          <Input
                            type="file"
                            accept=".srt,.ass,.ssa"
                            onChange={(e) => setSubtitleFile(e.target.files?.[0] || null)}
                            className="flex-1"
                          />
                          {subtitleFile && (
                            <span className="text-sm text-muted-foreground">
                              {subtitleFile.name}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Voice Mappings */}
                  <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                    <div className="mb-4 flex items-center justify-between">
                      <h3 className="font-semibold">角色声音映射 *</h3>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Users className="h-4 w-4" />
                        已映射 {voiceMappings.length} 个角色
                      </div>
                    </div>

                    {/* Add Mapping Form */}
                    <div className="mb-4 rounded-lg bg-gray-50 p-4 dark:bg-gray-900">
                      <div className="grid gap-3 sm:grid-cols-3">
                        <div>
                          <Label>角色名</Label>
                          <Input
                            placeholder="例如：主角"
                            value={newCharacter}
                            onChange={(e) => setNewCharacter(e.target.value)}
                          />
                        </div>
                        <div>
                          <Label>声音 ID</Label>
                          <select
                            value={newVoiceId}
                            onChange={(e) => setNewVoiceId(e.target.value)}
                            className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                          >
                            <option value="">选择声音...</option>
                            {voices.map((voice) => (
                              <option key={voice.id} value={voice.id}>
                                {voice.name} ({voice.id})
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="flex items-end">
                          <Button
                            onClick={handleAddMapping}
                            disabled={!newCharacter || !newVoiceId}
                            variant="outline"
                            className="w-full"
                          >
                            <Plus className="mr-2 h-4 w-4" />
                            添加
                          </Button>
                        </div>
                      </div>
                    </div>

                    {/* Mappings List */}
                    {voiceMappings.length > 0 && (
                      <div className="space-y-2">
                        {voiceMappings.map((mapping, index) => (
                          <div
                            key={index}
                            className="flex items-center justify-between rounded-lg border p-3"
                          >
                            <div className="flex items-center gap-3">
                              <div className="rounded-full bg-blue-100 px-3 py-1 text-sm font-medium text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
                                {mapping.character}
                              </div>
                              <span className="text-muted-foreground">→</span>
                              <span className="text-sm">{mapping.voiceId}</span>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleRemoveMapping(index)}
                            >
                              <Trash2 className="h-4 w-4 text-red-500" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    )}

                    {voiceMappings.length === 0 && (
                      <div className="py-4 text-center text-sm text-muted-foreground">
                        还没有添加角色映射
                      </div>
                    )}
                  </div>

                  {/* Submit Button */}
                  <Button
                    onClick={handleCreateJob}
                    disabled={creating || !jobName || !subtitleFile || voiceMappings.length === 0}
                    className="w-full"
                    size="lg"
                  >
                    {creating ? (
                      <>
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                        创建中...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-5 w-5" />
                        创建批量配音任务
                      </>
                    )}
                  </Button>
                </TabsContent>

                <TabsContent value="help" className="space-y-6">
                  <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                    <h3 className="mb-4 font-semibold">使用说明</h3>
                    <div className="space-y-4 text-sm">
                      <div>
                        <h4 className="font-medium">1. 准备字幕文件</h4>
                        <p className="mt-1 text-muted-foreground">
                          支持 SRT、ASS、SSA 格式的字幕文件。确保字幕文件包含对话内容和时间轴信息。
                        </p>
                      </div>
                      <div>
                        <h4 className="font-medium">2. 设置角色声音映射</h4>
                        <p className="mt-1 text-muted-foreground">
                          为每个角色分配对应的声音 ID。声音需要在"语音克隆工作室"中预先创建。
                        </p>
                      </div>
                      <div>
                        <h4 className="font-medium">3. 创建任务</h4>
                        <p className="mt-1 text-muted-foreground">
                          填写任务名称并上传字幕文件后点击创建。系统会自动处理所有对话行。
                        </p>
                      </div>
                      <div>
                        <h4 className="font-medium">4. 下载结果</h4>
                        <p className="mt-1 text-muted-foreground">
                          任务完成后，按角色分类的音频文件会保存在输出目录中。
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                    <h3 className="mb-4 font-semibold">字幕格式示例</h3>
                    <pre className="rounded-lg bg-gray-100 p-4 text-xs dark:bg-gray-900">
{`1
00:00:01,000 --> 00:00:03,000
主角: 안녕하세요?

2
00:00:03,500 --> 00:00:06,000
配角: 만나서 반갑습니다!`}
                    </pre>
                  </div>
                </TabsContent>
              </Tabs>
            </div>

            {/* Jobs List Sidebar */}
            <div className="space-y-6">
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="font-semibold">任务列表</h3>
                  <Button variant="outline" size="sm" onClick={loadData}>
                    刷新
                  </Button>
                </div>

                {loading ? (
                  <div className="flex justify-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                  </div>
                ) : batchJobs.length === 0 ? (
                  <div className="py-8 text-center text-sm text-muted-foreground">
                    还没有批量任务
                  </div>
                ) : (
                  <div className="space-y-3">
                    {batchJobs.map((job) => (
                      <div
                        key={job.job_id}
                        className="rounded-lg border p-4"
                      >
                        <div className="mb-2 flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <h4 className="truncate font-semibold">{job.name}</h4>
                              {getStatusIcon(job.status)}
                            </div>
                            <p className="mt-1 text-xs text-muted-foreground">
                              {getStatusText(job.status)}
                            </p>
                          </div>
                          {job.status === "pending" && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleDeleteJob(job.job_id)}
                            >
                              <Trash2 className="h-4 w-4 text-red-500" />
                            </Button>
                          )}
                        </div>

                        {/* Progress Bar */}
                        {job.status === "running" && job.total_items > 0 && (
                          <div className="mb-2 space-y-1">
                            <Progress value={job.progress * 100} />
                            <div className="flex justify-between text-xs text-muted-foreground">
                              <span>{formatProgress(job.completed_items, job.total_items)}</span>
                              <span>
                                {job.completed_items} / {job.total_items}
                              </span>
                            </div>
                          </div>
                        )}

                        {/* Error Message */}
                        {job.status === "failed" && job.error_message && (
                          <div className="mb-2 rounded-lg bg-red-50 p-2 text-xs text-red-700 dark:bg-red-900/20 dark:text-red-400">
                            {job.error_message}
                          </div>
                        )}

                        {/* Timestamps */}
                        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                          <span>创建: {formatDate(job.created_at)}</span>
                        </div>

                        {/* Download Button for Completed Jobs */}
                        {job.status === "completed" && job.output_path && (
                          <Button
                            variant="outline"
                            size="sm"
                            className="mt-2 w-full"
                            onClick={() => alert(`下载路径: ${job.output_path}`)}
                          >
                            <Download className="mr-2 h-4 w-4" />
                            下载结果
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Available Voices Quick View */}
              <div className="rounded-2xl border bg-white p-6 shadow-sm dark:bg-gray-800">
                <h3 className="mb-4 font-semibold">可用声音 ({voices.length})</h3>
                {voices.length === 0 ? (
                  <div className="py-4 text-center text-sm text-muted-foreground">
                    暂无可用声音
                  </div>
                ) : (
                  <div className="space-y-2">
                    {voices.slice(0, 5).map((voice) => (
                      <div
                        key={voice.id}
                        className="flex items-center justify-between text-sm"
                      >
                        <span className="font-medium">{voice.name}</span>
                        <span className="text-muted-foreground">{voice.id}</span>
                      </div>
                    ))}
                    {voices.length > 5 && (
                      <div className="pt-2 text-center text-xs text-muted-foreground">
                        还有 {voices.length - 5} 个声音...
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
