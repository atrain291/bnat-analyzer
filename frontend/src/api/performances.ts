import api from "./client";

export interface UploadResponse {
  performance_id: number;
  task_id: string;
  status: string;
}

export interface PipelineProgress {
  stage: string;
  pct: number;
  frame?: number;
  total_frames?: number;
  gpu_mem_pct?: number;
  cpu_pct?: number;
}

export interface PerformanceStatus {
  id: number;
  status: string;
  pipeline_progress: PipelineProgress | null;
  error: string | null;
}

export interface FrameData {
  id: number;
  timestamp_ms: number;
  dancer_pose: Record<string, { x: number; y: number; z: number; confidence: number }>;
  performance_dancer_id: number | null;
}

export interface AnalysisData {
  id: number;
  performance_dancer_id: number | null;
  aramandi_score: number | null;
  upper_body_score: number | null;
  symmetry_score: number | null;
  rhythm_consistency_score: number | null;
  overall_score: number | null;
  technique_scores: Record<string, number> | null;
  llm_summary: string | null;
  practice_plan: Record<string, unknown> | null;
  created_at: string;
}

export interface DetectedPerson {
  id: number;
  track_id: number;
  bbox: { x_min: number; y_min: number; x_max: number; y_max: number };
  representative_pose: Record<string, { x: number; y: number; z: number; confidence: number }>;
  frame_count: number;
  area: number;
}

export interface PerformanceDancer {
  id: number;
  track_id: number;
  label: string | null;
}

export interface DancerSelection {
  track_id: number;
  label?: string;
}

export interface Performance {
  id: number;
  dancer_id: number;
  item_name: string | null;
  item_type: string | null;
  talam: string | null;
  ragam: string | null;
  video_url: string | null;
  duration_ms: number | null;
  status: string;
  pipeline_progress: PipelineProgress | null;
  error: string | null;
  created_at: string;
  detection_frame_url: string | null;
  frames: FrameData[];
  analysis: AnalysisData[];
  detected_persons: DetectedPerson[];
  performance_dancers: PerformanceDancer[];
}

export async function uploadVideo(
  file: File,
  dancerId: number,
  itemName?: string,
  itemType?: string,
  talam?: string,
  ragam?: string,
  onProgress?: (pct: number) => void,
  signal?: AbortSignal
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("dancer_id", String(dancerId));
  if (itemName) form.append("item_name", itemName);
  if (itemType) form.append("item_type", itemType);
  if (talam) form.append("talam", talam);
  if (ragam) form.append("ragam", ragam);

  const { data } = await api.post("/upload/", form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
    },
    signal,
  });
  return data;
}

export async function getPerformanceStatus(id: number): Promise<PerformanceStatus> {
  const { data } = await api.get(`/performances/${id}/status`);
  return data;
}

export async function getPerformance(id: number): Promise<Performance> {
  const { data } = await api.get(`/performances/${id}`);
  return data;
}

export async function stopPerformance(id: number): Promise<{ status: string; message: string }> {
  const { data } = await api.post(`/performances/${id}/stop`);
  return data;
}

export async function deletePerformance(id: number): Promise<void> {
  await api.delete(`/performances/${id}`);
}

export async function getDetectedPersons(id: number): Promise<DetectedPerson[]> {
  const { data } = await api.get(`/performances/${id}/detected-persons`);
  return data;
}

export async function selectDancers(
  id: number,
  selections: DancerSelection[]
): Promise<{ task_id: string }> {
  const { data } = await api.post(`/performances/${id}/select-dancers`, { selections });
  return data;
}
