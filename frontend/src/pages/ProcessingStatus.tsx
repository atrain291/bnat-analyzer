import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Loader2, CheckCircle2, XCircle } from "lucide-react";
import { getPerformanceStatus, deletePerformance, stopPerformance, retryPerformance, PipelineProgress } from "../api/performances";

const STAGES = [
  { key: "tracking", label: "Tracking Dancers", weight: 30 },
  { key: "ingest", label: "Ingest Video", weight: 3 },
  { key: "beat_detection", label: "Beat Detection", weight: 4 },
  { key: "pose_estimation", label: "Pose Estimation", weight: 35 },
  { key: "pose_analysis", label: "Pose Analysis", weight: 5 },
  { key: "llm_synthesis", label: "AI Coaching", weight: 10 },
  { key: "scoring", label: "Scoring", weight: 3 },
  { key: "complete", label: "Complete", weight: 10 },
];

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

export default function ProcessingStatus() {
  const { performanceId } = useParams<{ performanceId: string }>();
  const navigate = useNavigate();
  const [status, setStatus] = useState<string>("queued");
  const [progress, setProgress] = useState<PipelineProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stageStartTimes = useRef<Record<string, number>>({});
  const [now, setNow] = useState(Date.now());

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
        if (data.status === "uploaded") {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setTimeout(() => navigate(`/select-frame/${performanceId}`), 500);
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

  // Track when each stage starts
  useEffect(() => {
    if (progress?.stage && !stageStartTimes.current[progress.stage]) {
      stageStartTimes.current[progress.stage] = Date.now();
    }
  }, [progress?.stage]);

  // Tick every second to update elapsed times
  useEffect(() => {
    if (status === "complete" || status === "failed") return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [status]);

  const handleCancel = async () => {
    if (!performanceId) return;
    await deletePerformance(Number(performanceId));
    navigate("/");
  };

  const [stopping, setStopping] = useState(false);

  const handleStop = async () => {
    if (!performanceId) return;
    setStopping(true);
    await stopPerformance(Number(performanceId));
    // Pipeline will finish analysis on collected frames, then polling will redirect
  };

  const currentStageIdx = STAGES.findIndex((s) => s.key === progress?.stage);

  return (
    <div className="mx-auto max-w-2xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Processing Video</h1>
        <p className="mt-1 text-gray-400">
          {status === "complete"
            ? "Analysis complete! Redirecting..."
            : status === "uploaded"
            ? "Redirecting to dancer selection..."
            : status === "failed"
            ? "Processing failed"
            : status === "tracking"
            ? "Tracking dancers through the video..."
            : stopping
            ? "Wrapping up with collected frames..."
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
            const weightStart = STAGES.slice(0, idx).reduce((s, st) => s + st.weight, 0);
            const weightEnd = weightStart + stage.weight;
            stagePct = Math.min(100, ((progress.pct - weightStart) / (weightEnd - weightStart)) * 100);
          }

          const stageStart = stageStartTimes.current[stage.key];
          const elapsed = stageStart ? (isDone && !isActive
            ? stageStartTimes.current[STAGES[idx + 1]?.key] || now
            : now) - stageStart : 0;

          return (
            <div key={stage.key} className="rounded-lg bg-gray-800 p-4">
              <div className="flex items-center justify-between mb-1">
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
                <div className="flex items-center gap-3">
                  {isActive && progress?.frame !== undefined && (
                    <span className="text-xs text-gray-400">
                      Frame {progress.frame} / {progress.total_frames}
                    </span>
                  )}
                  {(isActive || isDone) && stageStart && elapsed > 0 && (
                    <span className="text-xs text-gray-500">
                      {formatElapsed(elapsed)}
                    </span>
                  )}
                </div>
              </div>
              {isActive && progress?.message && (
                <p className="text-xs text-gray-400 ml-[26px] mb-2">{progress.message}</p>
              )}
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
          <>
            <button
              className="rounded-lg bg-brand-600 px-6 py-2 text-white hover:bg-brand-700"
              onClick={async () => {
                if (!performanceId) return;
                await retryPerformance(Number(performanceId));
                setStatus("processing");
                setError(null);
                stageStartTimes.current = {};
                intervalRef.current = setInterval(async () => {
                  const data = await getPerformanceStatus(Number(performanceId));
                  setStatus(data.status);
                  setProgress(data.pipeline_progress);
                  setError(data.error);
                  if (data.status === "complete") {
                    if (intervalRef.current) clearInterval(intervalRef.current);
                    setTimeout(() => navigate(`/review/${performanceId}`), 1500);
                  }
                  if (data.status === "failed") {
                    if (intervalRef.current) clearInterval(intervalRef.current);
                  }
                }, 2000);
              }}
            >
              Retry
            </button>
            <button
              className="rounded-lg bg-gray-700 px-6 py-2 text-white hover:bg-gray-600"
              onClick={() => navigate("/")}
            >
              Back to Dashboard
            </button>
          </>
        )}
        {(status === "processing" || status === "tracking") && (
          <button
            className="rounded-lg bg-brand-600 px-6 py-2 text-white hover:bg-brand-700 disabled:opacity-50"
            onClick={handleStop}
            disabled={stopping}
          >
            {stopping ? "Finishing analysis..." : "Stop & Analyze"}
          </button>
        )}
        {(status === "queued" || status === "processing" || status === "tracking") && (
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
