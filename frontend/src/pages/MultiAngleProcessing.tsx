import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Loader2, CheckCircle2, XCircle, Camera } from "lucide-react";
import { getMultiAngleGroupStatus, deleteMultiAngleGroup, GroupStatus } from "../api/multiAngle";

export default function MultiAngleProcessing() {
  const { groupId } = useParams<{ groupId: string }>();
  const navigate = useNavigate();
  const [groupStatus, setGroupStatus] = useState<GroupStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!groupId) return;

    const poll = async () => {
      try {
        const data = await getMultiAngleGroupStatus(Number(groupId));
        setGroupStatus(data);

        // All videos awaiting dancer selection -> go to cross-view linking
        if (data.all_awaiting_selection) {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setTimeout(() => navigate(`/multi-angle/link-dancers/${groupId}`), 500);
        }

        // All complete (individual pipelines done + fusion done)
        if (data.all_complete && data.group_status === "complete") {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setTimeout(() => navigate(`/multi-angle/review/${groupId}`), 1500);
        }

        if (data.any_failed) {
          setError("One or more video pipelines failed.");
        }
      } catch {
        // ignore polling errors
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 2000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [groupId, navigate]);

  const handleCancel = async () => {
    if (!groupId) return;
    await deleteMultiAngleGroup(Number(groupId));
    navigate("/");
  };

  return (
    <div className="mx-auto max-w-2xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Processing Multi-Angle Videos</h1>
        <p className="mt-1 text-gray-400">
          {groupStatus?.all_awaiting_selection
            ? "All videos scanned! Redirecting to dancer linking..."
            : groupStatus?.all_complete && groupStatus.group_status === "complete"
            ? "Multi-angle analysis complete! Redirecting..."
            : groupStatus?.group_status === "processing"
            ? "Analyzing videos from each angle..."
            : "Scanning videos for dancers..."}
        </p>
      </div>

      {/* Per-video status */}
      <div className="space-y-3">
        {groupStatus?.performances.map((p) => {
          const isDone = p.status === "complete" || p.status === "awaiting_selection";
          const isFailed = p.status === "failed";
          const pct = p.pipeline_progress?.pct ?? 0;

          return (
            <div key={p.performance_id} className="rounded-lg bg-gray-800 p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {isDone ? (
                    <CheckCircle2 size={18} className="text-green-400" />
                  ) : isFailed ? (
                    <XCircle size={18} className="text-red-400" />
                  ) : (
                    <Loader2 size={18} className="text-brand-400 animate-spin" />
                  )}
                  <Camera size={16} className="text-gray-400" />
                  <span className="text-white font-medium">
                    {p.camera_label || `Camera`}
                  </span>
                </div>
                <span className="text-xs text-gray-400 capitalize">
                  {p.status === "awaiting_selection" ? "scanned" : p.status.replace("_", " ")}
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-gray-700 overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    isFailed ? "bg-red-500" : isDone ? "bg-green-500" : "bg-brand-500"
                  }`}
                  style={{ width: `${isDone ? 100 : pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Fusion status (after individual pipelines) */}
      {groupStatus?.group_status === "processing" && groupStatus.all_complete && (
        <div className="rounded-lg bg-gray-800 p-4">
          <div className="flex items-center gap-2">
            <Loader2 size={18} className="text-brand-400 animate-spin" />
            <span className="text-white font-medium">Fusing multi-angle scores...</span>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-red-900/30 border border-red-700 p-4 text-red-300">
          {error}
        </div>
      )}

      <div className="flex justify-center">
        <button
          className="rounded-lg bg-red-800 px-6 py-2 text-white hover:bg-red-700"
          onClick={handleCancel}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
