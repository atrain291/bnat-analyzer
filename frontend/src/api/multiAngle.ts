import api from "./client";
import type { UploadResponse, Performance } from "./performances";

export interface MultiAngleGroupListItem {
  id: number;
  dancer_id: number;
  item_name: string | null;
  status: string;
  created_at: string;
  performance_count: number;
  overall_score: number | null;
}

export interface MultiAngleAnalysis {
  id: number;
  dancer_label: string | null;
  aramandi_score: number | null;
  upper_body_score: number | null;
  symmetry_score: number | null;
  rhythm_consistency_score: number | null;
  overall_score: number | null;
  per_view_scores: Record<number, Record<string, number | null>> | null;
  score_sources: Record<string, number | string> | null;
  llm_summary: string | null;
  created_at: string;
}

export interface MultiAngleGroup {
  id: number;
  dancer_id: number;
  item_name: string | null;
  item_type: string | null;
  talam: string | null;
  ragam: string | null;
  sync_offsets: Record<string, number> | null;
  sync_confidence: number | null;
  status: string;
  created_at: string;
  performances: Performance[];
  multi_angle_analyses: MultiAngleAnalysis[];
}

export interface MultiAngleUploadResponse {
  group_id: number;
  performances: UploadResponse[];
}

export interface GroupStatus {
  group_id: number;
  group_status: string;
  all_awaiting_selection: boolean;
  any_failed: boolean;
  all_complete: boolean;
  performances: {
    performance_id: number;
    camera_label: string | null;
    status: string;
    pipeline_progress: { stage: string; pct: number } | null;
  }[];
}

export interface CrossViewDancerLink {
  label: string;
  performance_tracks: Record<number, number>;
}

export async function uploadMultiAngle(
  files: File[],
  dancerId: number,
  cameraLabels: string[],
  itemName?: string,
  itemType?: string,
  talam?: string,
  ragam?: string,
  onProgress?: (pct: number) => void,
  signal?: AbortSignal,
): Promise<MultiAngleUploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("dancer_id", String(dancerId));
  form.append("camera_labels", cameraLabels.join(","));
  if (itemName) form.append("item_name", itemName);
  if (itemType) form.append("item_type", itemType);
  if (talam) form.append("talam", talam);
  if (ragam) form.append("ragam", ragam);

  const { data } = await api.post("/multi-angle/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
    },
    signal,
  });
  return data;
}

export async function listMultiAngleGroups(dancerId?: number): Promise<MultiAngleGroupListItem[]> {
  const params = dancerId ? { dancer_id: dancerId } : {};
  const { data } = await api.get("/multi-angle/groups", { params });
  return data;
}

export async function getMultiAngleGroup(groupId: number): Promise<MultiAngleGroup> {
  const { data } = await api.get(`/multi-angle/groups/${groupId}`);
  return data;
}

export async function getMultiAngleGroupStatus(groupId: number): Promise<GroupStatus> {
  const { data } = await api.get(`/multi-angle/groups/${groupId}/status`);
  return data;
}

export async function linkDancersAcrossViews(
  groupId: number,
  links: CrossViewDancerLink[],
): Promise<{ status: string; fusion_task_id: string }> {
  const { data } = await api.post(`/multi-angle/groups/${groupId}/link-dancers`, { links });
  return data;
}

export async function deleteMultiAngleGroup(groupId: number): Promise<void> {
  await api.delete(`/multi-angle/groups/${groupId}`);
}
