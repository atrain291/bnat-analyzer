import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, Camera, Play, Pause, SkipBack } from "lucide-react";
import { getMultiAngleGroup, MultiAngleGroup } from "../api/multiAngle";
import { getPerformanceFrames, FrameData } from "../api/performances";

const SKELETON_CONNECTIONS: [string, string][] = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["nose", "left_eye"],
  ["nose", "right_eye"],
  ["nose", "left_shoulder"],
  ["nose", "right_shoulder"],
];

const PLAYBACK_SPEEDS = [0.25, 0.5, 1, 1.5, 2];
const SAFFRON = "#F9A825";

interface ViewState {
  performanceId: number;
  cameraLabel: string;
  videoUrl: string;
  syncOffset: number; // ms relative to reference
  frames: FrameData[];
}

export default function MultiAngleReview() {
  const { groupId } = useParams<{ groupId: string }>();
  const navigate = useNavigate();
  const [group, setGroup] = useState<MultiAngleGroup | null>(null);
  const [views, setViews] = useState<ViewState[]>([]);
  const [loading, setLoading] = useState(true);

  // Playback state (shared across views)
  const [playing, setPlaying] = useState(false);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);
  const [duration, setDuration] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [showSkeleton, setShowSkeleton] = useState(true);

  const videoRefs = useRef<(HTMLVideoElement | null)[]>([]);
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const animRef = useRef<number>(0);

  const apiBase = import.meta.env.VITE_API_URL || "http://localhost:8000";

  useEffect(() => {
    if (!groupId) return;
    setLoading(true);

    getMultiAngleGroup(Number(groupId))
      .then(async (g) => {
        setGroup(g);

        const syncOffsets = g.sync_offsets || {};
        const viewStates: ViewState[] = [];

        for (const perf of g.performances) {
          const frames = await getPerformanceFrames(perf.id);
          viewStates.push({
            performanceId: perf.id,
            cameraLabel: perf.camera_label || "Camera",
            videoUrl: perf.video_url || "",
            syncOffset: syncOffsets[String(perf.id)] || 0,
            frames,
          });
        }

        setViews(viewStates);

        // Set duration to longest video
        const maxDuration = Math.max(...g.performances.map((p) => p.duration_ms || 0));
        setDuration(maxDuration);
      })
      .catch(() => navigate("/"))
      .finally(() => setLoading(false));
  }, [groupId, navigate]);

  // Synchronized playback loop
  const tick = useCallback(() => {
    const primaryVideo = videoRefs.current[0];
    if (!primaryVideo) return;

    const timeMs = primaryVideo.currentTime * 1000;
    setCurrentTimeMs(timeMs);

    // Sync other videos
    for (let i = 1; i < videoRefs.current.length; i++) {
      const vid = videoRefs.current[i];
      const view = views[i];
      if (!vid || !view) continue;

      const targetTime = (timeMs - view.syncOffset) / 1000;
      if (Math.abs(vid.currentTime - targetTime) > 0.1) {
        vid.currentTime = Math.max(0, targetTime);
      }
    }

    // Draw skeletons
    if (showSkeleton) {
      views.forEach((view, idx) => {
        drawSkeleton(idx, timeMs - view.syncOffset);
      });
    }

    if (playing) {
      animRef.current = requestAnimationFrame(tick);
    }
  }, [playing, views, showSkeleton]);

  useEffect(() => {
    if (playing) {
      animRef.current = requestAnimationFrame(tick);
    }
    return () => cancelAnimationFrame(animRef.current);
  }, [playing, tick]);

  const togglePlay = () => {
    videoRefs.current.forEach((v) => {
      if (!v) return;
      if (playing) v.pause();
      else v.play();
    });
    setPlaying(!playing);
  };

  const restart = () => {
    videoRefs.current.forEach((v, i) => {
      if (!v) return;
      const offset = views[i]?.syncOffset || 0;
      v.currentTime = Math.max(0, -offset / 1000);
    });
    setCurrentTimeMs(0);
  };

  const seek = (timeMs: number) => {
    videoRefs.current.forEach((v, i) => {
      if (!v) return;
      const offset = views[i]?.syncOffset || 0;
      v.currentTime = Math.max(0, (timeMs - offset) / 1000);
    });
    setCurrentTimeMs(timeMs);

    if (showSkeleton) {
      views.forEach((view, idx) => {
        drawSkeleton(idx, timeMs - view.syncOffset);
      });
    }
  };

  const changeSpeed = () => {
    const idx = PLAYBACK_SPEEDS.indexOf(speed);
    const next = PLAYBACK_SPEEDS[(idx + 1) % PLAYBACK_SPEEDS.length];
    setSpeed(next);
    videoRefs.current.forEach((v) => {
      if (v) v.playbackRate = next;
    });
  };

  const drawSkeleton = (viewIdx: number, localTimeMs: number) => {
    const canvas = canvasRefs.current[viewIdx];
    const video = videoRefs.current[viewIdx];
    const view = views[viewIdx];
    if (!canvas || !video || !view) return;

    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!showSkeleton || !view.frames.length) return;

    // Find closest frame
    let closest = view.frames[0];
    let minDist = Infinity;
    for (const frame of view.frames) {
      const dist = Math.abs(frame.timestamp_ms - localTimeMs);
      if (dist < minDist) {
        minDist = dist;
        closest = frame;
      }
    }

    if (minDist > 200) return; // too far from any frame

    const pose = closest.dancer_pose;
    if (!pose) return;

    const w = canvas.width;
    const h = canvas.height;

    // Skeleton lines
    ctx.lineWidth = 3;
    ctx.strokeStyle = SAFFRON;
    ctx.globalAlpha = 0.9;

    for (const [from, to] of SKELETON_CONNECTIONS) {
      const a = pose[from];
      const b = pose[to];
      if (!a || !b || a.confidence < 0.3 || b.confidence < 0.3) continue;
      ctx.beginPath();
      ctx.moveTo(a.x * w, a.y * h);
      ctx.lineTo(b.x * w, b.y * h);
      ctx.stroke();
    }

    // Keypoints
    for (const kp of Object.values(pose) as { x: number; y: number; confidence: number }[]) {
      if (kp.confidence < 0.3) continue;
      ctx.fillStyle = SAFFRON;
      ctx.beginPath();
      ctx.arc(kp.x * w, kp.y * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1.0;
  };

  if (loading) {
    return <div className="text-center text-gray-400 py-20">Loading multi-angle review...</div>;
  }

  if (!group) return null;

  const fusedAnalyses = group.multi_angle_analyses || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={() => navigate("/")} className="text-gray-400 hover:text-white">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-white">
            {group.item_name || group.item_type || "Multi-Angle Review"}
          </h1>
          <p className="text-gray-400 text-sm">
            {views.length} camera angles
            {group.sync_confidence !== null && ` | Sync confidence: ${(group.sync_confidence * 100).toFixed(0)}%`}
          </p>
        </div>
      </div>

      {/* Synchronized video players */}
      <div className={`grid gap-4 ${views.length === 2 ? "grid-cols-2" : "grid-cols-1 lg:grid-cols-2"}`}>
        {views.map((view, idx) => (
          <div key={view.performanceId} className="rounded-lg overflow-hidden bg-black relative">
            <div className="absolute top-2 left-2 z-10 bg-black/70 rounded px-2 py-1 text-xs text-white flex items-center gap-1">
              <Camera size={12} /> {view.cameraLabel}
              {view.syncOffset !== 0 && (
                <span className="text-gray-400 ml-1">
                  ({view.syncOffset > 0 ? "+" : ""}{view.syncOffset}ms)
                </span>
              )}
            </div>
            <video
              ref={(el) => { videoRefs.current[idx] = el; }}
              src={`${apiBase}${view.videoUrl}`}
              className="w-full"
              muted={idx > 0} // Only first video has audio
              playsInline
              preload="auto"
              onTimeUpdate={() => {
                if (!playing && idx === 0) {
                  const timeMs = (videoRefs.current[0]?.currentTime || 0) * 1000;
                  setCurrentTimeMs(timeMs);
                }
              }}
            />
            <canvas
              ref={(el) => { canvasRefs.current[idx] = el; }}
              className="absolute top-0 left-0 w-full h-full pointer-events-none"
            />
          </div>
        ))}
      </div>

      {/* Shared playback controls */}
      <div className="rounded-lg bg-gray-800 p-4 space-y-3">
        {/* Seek bar */}
        <input
          type="range"
          min={0}
          max={duration}
          value={currentTimeMs}
          onChange={(e) => seek(Number(e.target.value))}
          className="w-full accent-brand-500"
        />
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button onClick={restart} className="text-gray-400 hover:text-white">
              <SkipBack size={20} />
            </button>
            <button onClick={togglePlay} className="text-white hover:text-brand-400">
              {playing ? <Pause size={24} /> : <Play size={24} />}
            </button>
            <button
              onClick={changeSpeed}
              className="rounded bg-gray-700 px-2 py-1 text-xs text-gray-300 hover:bg-gray-600"
            >
              {speed}x
            </button>
          </div>
          <div className="flex items-center gap-3 text-sm text-gray-400">
            <span>
              {(currentTimeMs / 1000).toFixed(1)}s / {(duration / 1000).toFixed(1)}s
            </span>
            <button
              onClick={() => setShowSkeleton(!showSkeleton)}
              className={`rounded px-2 py-1 text-xs ${
                showSkeleton ? "bg-brand-600 text-white" : "bg-gray-700 text-gray-400"
              }`}
            >
              Skeleton
            </button>
          </div>
        </div>
      </div>

      {/* Fused multi-angle scores */}
      {fusedAnalyses.length > 0 && (
        <section className="rounded-lg bg-gray-800 p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white">Multi-Angle Fused Scores</h2>
          {fusedAnalyses.map((analysis) => (
            <div key={analysis.id} className="space-y-4">
              {analysis.dancer_label && (
                <h3 className="text-brand-400 font-medium">{analysis.dancer_label}</h3>
              )}

              {/* Score cards */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                {[
                  { label: "Overall", value: analysis.overall_score },
                  { label: "Aramandi", value: analysis.aramandi_score },
                  { label: "Upper Body", value: analysis.upper_body_score },
                  { label: "Symmetry", value: analysis.symmetry_score },
                  { label: "Rhythm", value: analysis.rhythm_consistency_score },
                ].map((s) => (
                  <div key={s.label} className="rounded-lg bg-gray-700/50 p-3 text-center">
                    <div className="text-2xl font-bold text-brand-400">
                      {s.value !== null ? s.value : "—"}
                    </div>
                    <div className="text-xs text-gray-400">{s.label}</div>
                  </div>
                ))}
              </div>

              {/* Per-view score comparison */}
              {analysis.per_view_scores && (
                <div className="space-y-2">
                  <h4 className="text-sm text-gray-400 font-medium">Per-View Breakdown</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-gray-400 border-b border-gray-700">
                          <th className="text-left py-1 pr-4">View</th>
                          <th className="text-right py-1 px-2">Aramandi</th>
                          <th className="text-right py-1 px-2">Upper Body</th>
                          <th className="text-right py-1 px-2">Symmetry</th>
                          <th className="text-right py-1 px-2">Overall</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(analysis.per_view_scores).map(([perfId, scores]) => {
                          const view = views.find((v) => v.performanceId === Number(perfId));
                          return (
                            <tr key={perfId} className="text-gray-300 border-b border-gray-700/50">
                              <td className="py-1 pr-4 flex items-center gap-1">
                                <Camera size={12} className="text-gray-500" />
                                {view?.cameraLabel || `Camera ${perfId}`}
                              </td>
                              <td className="text-right py-1 px-2">{scores.aramandi_score ?? "—"}</td>
                              <td className="text-right py-1 px-2">{scores.upper_body_score ?? "—"}</td>
                              <td className="text-right py-1 px-2">{scores.symmetry_score ?? "—"}</td>
                              <td className="text-right py-1 px-2 font-medium">{scores.overall_score ?? "—"}</td>
                            </tr>
                          );
                        })}
                        <tr className="text-brand-400 font-medium">
                          <td className="py-1 pr-4">Fused</td>
                          <td className="text-right py-1 px-2">{analysis.aramandi_score ?? "—"}</td>
                          <td className="text-right py-1 px-2">{analysis.upper_body_score ?? "—"}</td>
                          <td className="text-right py-1 px-2">{analysis.symmetry_score ?? "—"}</td>
                          <td className="text-right py-1 px-2">{analysis.overall_score ?? "—"}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Multi-angle coaching */}
              {analysis.llm_summary && (
                <div className="rounded-lg bg-gray-700/30 p-4">
                  <h4 className="text-sm text-gray-400 font-medium mb-2">Multi-Angle Coaching</h4>
                  <div className="text-gray-300 text-sm whitespace-pre-wrap leading-relaxed">
                    {analysis.llm_summary}
                  </div>
                </div>
              )}
            </div>
          ))}
        </section>
      )}

      {/* Individual view analyses (collapsed) */}
      {group.performances.some((p) => p.analysis?.length > 0) && (
        <section className="rounded-lg bg-gray-800 p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white">Individual View Analyses</h2>
          <p className="text-sm text-gray-400">
            Each camera angle was also analyzed independently.
            Click to view the full analysis for each angle.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {group.performances.map((perf) => (
              <button
                key={perf.id}
                className="rounded-lg bg-gray-700/50 p-4 text-left hover:bg-gray-700 transition-colors"
                onClick={() => navigate(`/review/${perf.id}`)}
              >
                <div className="flex items-center gap-2 text-white font-medium">
                  <Camera size={16} className="text-brand-400" />
                  {perf.camera_label || "Camera"}
                </div>
                {perf.analysis?.[0] && (
                  <div className="mt-1 text-sm text-gray-400">
                    Overall: <span className="text-brand-400 font-medium">{perf.analysis[0].overall_score}</span>
                  </div>
                )}
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
