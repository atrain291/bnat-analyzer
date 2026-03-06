import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Loader2, CheckCircle2, XCircle } from "lucide-react";
import { getPerformanceStatus, deletePerformance, stopPerformance, PipelineProgress } from "../api/performances";

const STAGES = [
  { key: "ingest", label: "Ingest Video", weight: 3 },
  { key: "detection", label: "Detecting Dancers", weight: 7 },
  { key: "pose_estimation", label: "Pose Estimation", weight: 55 },
  { key: "pose_analysis", label: "Pose Analysis", weight: 5 },
  { key: "llm_synthesis", label: "AI Coaching", weight: 15 },
  { key: "scoring", label: "Scoring", weight: 5 },
  { key: "complete", label: "Complete", weight: 10 },
];

export default function ProcessingStatus() {
  const { performanceId } = useParams<{ performanceId: string }>();
  const navigate = useNavigate();
  const [status, setStatus] = useState<string>("queued");
  const [progress, setProgress] = useState<PipelineProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!performanceId) return;

    const poll = async () => {
      try {
        const data = await getPerformanceStatus(Number(performanceId));
        setStatus(data.status);
        setProgress(data.pipeline_progress);
        setError(data.error);

        if (data.status === "complete") {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setTimeout(() => navigate(`/review/${performanceId}`), 1500);
        }
        if (data.status === "awaiting_selection") {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setTimeout(() => navigate(`/select-dancers/${performanceId}`), 500);
        }
        if (data.status === "failed") {
          if (intervalRef.current) clearInterval(intervalRef.current);
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
  }, [performanceId, navigate]);

  const handleCancel = async () => {
    if (!performanceId) return;
    await deletePerformance(Number(performanceId));
    navigate("/");
  };

  const handleStop = async () => {
    if (!performanceId) return;
    await stopPerformance(Number(performanceId));
    if (intervalRef.current) clearInterval(intervalRef.current);
    navigate(`/review/${performanceId}`);
  };

  const currentStageIdx = STAGES.findIndex((s) => s.key === progress?.stage);

  return (
    <div className="mx-auto max-w-2xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Processing Video</h1>
        <p className="mt-1 text-gray-400">
          {status === "complete"
            ? "Analysis complete! Redirecting..."
            : status === "awaiting_selection"
            ? "Dancers detected! Redirecting to selection..."
            : status === "failed"
            ? "Processing failed"
            : status === "detecting"
            ? "Scanning video for dancers..."
            : "Analyzing your Bharatanatyam performance..."}
        </p>
      </div>

      {/* Pipeline Stages */}
      <div className="space-y-4">
        {STAGES.map((stage, idx) => {
          const isActive = stage.key === progress?.stage;
          const isDone = currentStageIdx > idx || status === "complete";
          const isFailed = status === "failed" && isActive;

          let stagePct = 0;
          if (isDone) stagePct = 100;
          else if (isActive && progress) {
            const stageStart = STAGES.slice(0, idx).reduce((s, st) => s + st.weight, 0);
            const stageEnd = stageStart + stage.weight;
            stagePct = Math.min(100, ((progress.pct - stageStart) / (stageEnd - stageStart)) * 100);
          }

          return (
            <div key={stage.key} className="rounded-lg bg-gray-800 p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {isDone ? (
                    <CheckCircle2 size={18} className="text-green-400" />
                  ) : isFailed ? (
                    <XCircle size={18} className="text-red-400" />
                  ) : isActive ? (
                    <Loader2 size={18} className="text-brand-400 animate-spin" />
                  ) : (
                    <div className="w-[18px] h-[18px] rounded-full border-2 border-gray-600" />
                  )}
                  <span className={isActive ? "text-white font-medium" : "text-gray-400"}>
                    {stage.label}
                  </span>
                </div>
                {isActive && progress?.frame !== undefined && (
                  <span className="text-xs text-gray-400">
                    Frame {progress.frame} / {progress.total_frames}
                  </span>
                )}
              </div>
              <div className="h-1.5 rounded-full bg-gray-700 overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    isFailed ? "bg-red-500" : isDone ? "bg-green-500" : "bg-brand-500"
                  }`}
                  style={{ width: `${stagePct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Overall progress */}
      {progress && (
        <div className="text-center text-sm text-gray-400">
          Overall: {progress.pct.toFixed(1)}%
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-lg bg-red-900/30 border border-red-700 p-4 text-red-300">
          {error}
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-center gap-4">
        {status === "failed" && (
          <button
            className="rounded-lg bg-gray-700 px-6 py-2 text-white hover:bg-gray-600"
            onClick={() => navigate("/")}
          >
            Back to Dashboard
          </button>
        )}
        {status === "processing" && (
          <button
            className="rounded-lg bg-brand-600 px-6 py-2 text-white hover:bg-brand-700"
            onClick={handleStop}
          >
            Stop &amp; Review
          </button>
        )}
        {(status === "queued" || status === "processing" || status === "detecting") && (
          <button
            className="rounded-lg bg-red-800 px-6 py-2 text-white hover:bg-red-700"
            onClick={handleCancel}
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}
