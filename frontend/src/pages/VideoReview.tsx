import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { Play, Pause, SkipBack, Trash2, ArrowLeft } from "lucide-react";
import { getPerformance, getPerformanceFrames, deletePerformance, Performance, FrameData, AnalysisData } from "../api/performances";

// COCO skeleton connections for Bharatanatyam visualization
const SKELETON_CONNECTIONS: [string, string][] = [
  // Torso
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  // Left arm
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  // Right arm
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  // Left leg
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  // Right leg
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  // Head
  ["nose", "left_eye"],
  ["nose", "right_eye"],
  ["left_eye", "left_ear"],
  ["right_eye", "right_ear"],
  ["nose", "left_shoulder"],
  ["nose", "right_shoulder"],
];

const PLAYBACK_SPEEDS = [0.25, 0.5, 1, 1.5, 2];

const DANCER_COLORS = [
  "#F9A825", // saffron
  "#00BCD4", // teal
  "#E91E63", // magenta
  "#8BC34A", // lime
  "#FF7043", // coral
  "#7C4DFF", // violet
];

const SAFFRON = "#F9A825";
const SAFFRON_DIM = "rgba(249, 168, 37, 0.4)";

interface ScoreDetail {
  label: string;
  value: number | null;
  measured?: string;
  ideal?: string;
  tip?: string;
}

function getScoreDetails(analysis: AnalysisData): ScoreDetail[] {
  const inputs = analysis.technique_scores?.inputs as Record<string, number> | undefined;

  const fmt = (v: number | undefined, unit: string, decimals = 1) =>
    v !== undefined ? `${v.toFixed(decimals)}${unit}` : "N/A";

  return [
    {
      label: "Overall",
      value: analysis.overall_score,
      measured: "Weighted composite of all scores below",
      ideal: "100 = perfect across all dimensions",
      tip: "Focus on your lowest individual score for the biggest improvement.",
    },
    {
      label: "Aramandi",
      value: analysis.aramandi_score,
      measured: inputs ? `Avg knee angle: ${fmt(inputs.avg_knee_angle, "°")} (std: ${fmt(inputs.knee_angle_std, "°")})` : undefined,
      ideal: "105° knee bend with low variance",
      tip: inputs?.avg_knee_angle !== undefined
        ? inputs.avg_knee_angle > 140
          ? "Your knees are too straight. Bend deeper into aramandi — aim for a half-seated position with knees over toes."
          : inputs.avg_knee_angle < 80
          ? "You're bending too deep. Ease up slightly — aramandi should feel strong and sustainable, not a full squat."
          : inputs.knee_angle_std !== undefined && inputs.knee_angle_std > 15
          ? "Your aramandi depth is inconsistent. Practice holding a steady half-sit to build muscle memory."
          : "Good aramandi range. Maintain consistency throughout the piece."
        : undefined,
    },
    {
      label: "Upper Body",
      value: analysis.upper_body_score,
      measured: inputs ? `Avg torso deviation: ${fmt(inputs.avg_torso_angle, "°")}` : undefined,
      ideal: "0° deviation (perfectly upright)",
      tip: inputs?.avg_torso_angle !== undefined
        ? inputs.avg_torso_angle > 8
          ? "Your torso is leaning significantly. Engage your core and imagine a string pulling up from the crown of your head."
          : inputs.avg_torso_angle > 4
          ? "Slight torso lean detected. Check your posture in the mirror — even small leans are visible on stage."
          : "Excellent uprightness. Your torso alignment is strong."
        : undefined,
    },
    {
      label: "Symmetry",
      value: analysis.symmetry_score,
      measured: inputs ? `Hip asymmetry: ${fmt(inputs.hip_symmetry_avg, "", 3)}` : undefined,
      ideal: "0.0 hip asymmetry (perfectly balanced)",
      tip: inputs?.hip_symmetry_avg !== undefined
        ? inputs.hip_symmetry_avg > 0.10
          ? "Significant hip asymmetry — one side is higher/shifted. Practice in front of a mirror, checking your hip line is level."
          : inputs.hip_symmetry_avg > 0.05
          ? "Mild hip asymmetry. Focus on distributing weight evenly between both feet during aramandi."
          : "Good bilateral symmetry. Your alignment is balanced."
        : undefined,
    },
    {
      label: "Foot Technique",
      value: analysis.technique_scores?.foot_technique_score ?? null,
      measured: inputs
        ? `Avg turnout: ${fmt(inputs.avg_foot_turnout, "°")} | Flatness: ${fmt(inputs.avg_foot_flatness, "", 4)}`
        : undefined,
      ideal: "45–60° turnout, feet flat on floor",
      tip: inputs?.avg_foot_turnout !== undefined
        ? inputs.avg_foot_turnout > 70
          ? "Your feet are turned out too far, which can strain knees. Aim for a natural 45-60° turnout."
          : inputs.avg_foot_turnout < 30
          ? "Your feet need more turnout. Rotate from the hips, not just the ankles, for proper Bharatanatyam stance."
          : inputs.avg_foot_flatness !== undefined && inputs.avg_foot_flatness > 0.03
          ? "Your feet are lifting off the floor. In aramandi, keep both feet firmly grounded for powerful stamps."
          : "Good foot positioning. Maintain this turnout and grounding."
        : undefined,
    },
  ];
}

function ScoreCards({ analysis }: { analysis: AnalysisData }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const details = getScoreDetails(analysis);

  const hasScores = details.some((s) => s.value !== null);
  if (!hasScores) return null;

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {details.map(
          (s) =>
            s.value !== null && (
              <button
                key={s.label}
                className={`rounded-lg p-3 text-center transition-all ${
                  s.label === "Overall"
                    ? "bg-brand-600/20 border border-brand-500/30"
                    : "bg-gray-700"
                } ${expanded === s.label ? "ring-2 ring-brand-400" : "hover:ring-1 hover:ring-gray-500"}`}
                onClick={() => setExpanded(expanded === s.label ? null : s.label)}
              >
                <div className="text-2xl font-bold text-brand-400">
                  {s.value}
                  <span className="text-sm font-normal text-gray-500"> / 100</span>
                </div>
                <div className="text-xs text-gray-400">{s.label}</div>
              </button>
            )
        )}
      </div>
      {expanded && (() => {
        const detail = details.find((s) => s.label === expanded);
        if (!detail) return null;
        return (
          <div className="rounded-lg bg-gray-700/50 border border-gray-600 p-4 space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="font-medium text-white">{detail.label} Score Breakdown</span>
              <button className="text-gray-400 hover:text-white text-xs" onClick={() => setExpanded(null)}>close</button>
            </div>
            {detail.measured && (
              <div><span className="text-gray-400">Measured: </span><span className="text-gray-200">{detail.measured}</span></div>
            )}
            {detail.ideal && (
              <div><span className="text-gray-400">Ideal: </span><span className="text-gray-200">{detail.ideal}</span></div>
            )}
            {detail.tip && (
              <div className="mt-2 rounded bg-brand-600/10 border border-brand-500/20 px-3 py-2 text-brand-300">
                <span className="font-medium">Tip: </span>{detail.tip}
              </div>
            )}
          </div>
        );
      })()}
      <p className="text-xs text-gray-500 text-center">Click any score for details</p>
    </div>
  );
}

export default function VideoReview() {
  const { performanceId } = useParams<{ performanceId: string }>();
  const navigate = useNavigate();
  const [perf, setPerf] = useState<Performance | null>(null);
  const [frames, setFrames] = useState<FrameData[]>([]);
  const [loading, setLoading] = useState(true);
  const [framesLoading, setFramesLoading] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [activeDancerTab, setActiveDancerTab] = useState<number | null>(null);
  const [visibleDancers, setVisibleDancers] = useState<Set<number>>(new Set());

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animFrameRef = useRef<number>(0);

  useEffect(() => {
    if (!performanceId) return;
    const id = Number(performanceId);
    setLoading(true);
    setFramesLoading(true);

    // Load performance metadata (fast) and frames (slow) in parallel
    getPerformance(id)
      .then((data) => {
        setPerf(data);
        if (data.performance_dancers.length > 0) {
          const ids = new Set(data.performance_dancers.map((d) => d.id));
          setVisibleDancers(ids);
          setActiveDancerTab(data.performance_dancers[0].id);
        }
      })
      .catch(() => navigate("/"))
      .finally(() => setLoading(false));

    getPerformanceFrames(id)
      .then(setFrames)
      .catch(() => {})
      .finally(() => setFramesLoading(false));
  }, [performanceId, navigate]);

  // Find the closest frame for a given timestamp using binary search
  const findFrame = useCallback(
    (timeMs: number, frames: FrameData[]): FrameData | null => {
      if (!frames.length) return null;
      let lo = 0;
      let hi = frames.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (frames[mid].timestamp_ms < timeMs) lo = mid + 1;
        else hi = mid;
      }
      // Pick closest between lo and lo-1
      if (lo > 0) {
        const diffLo = Math.abs(frames[lo].timestamp_ms - timeMs);
        const diffPrev = Math.abs(frames[lo - 1].timestamp_ms - timeMs);
        if (diffPrev < diffLo) return frames[lo - 1];
      }
      return frames[lo];
    },
    []
  );

  const getDancerColor = useCallback(
    (dancerId: number | null): string => {
      if (!perf || !dancerId) return SAFFRON;
      const idx = perf.performance_dancers.findIndex((d) => d.id === dancerId);
      return idx >= 0 ? DANCER_COLORS[idx % DANCER_COLORS.length] : SAFFRON;
    },
    [perf]
  );

  const drawSkeletons = useCallback(
    (frames: FrameData[], canvas: HTMLCanvasElement) => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!showSkeleton) return;

      const w = canvas.width;
      const h = canvas.height;

      for (const frame of frames) {
        const pdId = frame.performance_dancer_id;
        if (pdId && !visibleDancers.has(pdId)) continue;

        const color = getDancerColor(pdId);
        const pose = frame.dancer_pose;
        if (!pose || Object.keys(pose).length === 0) continue;

        ctx.lineWidth = 3;
        for (const [from, to] of SKELETON_CONNECTIONS) {
          const a = pose[from];
          const b = pose[to];
          if (!a || !b || a.confidence < 0.3 || b.confidence < 0.3) continue;

          const alpha = Math.min(a.confidence, b.confidence);
          ctx.globalAlpha = alpha > 0.5 ? 1.0 : 0.4;
          ctx.strokeStyle = color;
          ctx.beginPath();
          ctx.moveTo(a.x * w, a.y * h);
          ctx.lineTo(b.x * w, b.y * h);
          ctx.stroke();
        }

        for (const [, kp] of Object.entries(pose)) {
          if (kp.confidence < 0.3) continue;
          ctx.globalAlpha = kp.confidence > 0.5 ? 1.0 : 0.4;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(kp.x * w, kp.y * h, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.globalAlpha = 1.0;
    },
    [showSkeleton, visibleDancers, getDancerColor]
  );

  // Animation loop
  useEffect(() => {
    if (!perf || !frames.length) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    // Group frames by dancer for multi-dancer rendering
    const framesByDancer = new Map<number | null, FrameData[]>();
    for (const f of frames) {
      const key = f.performance_dancer_id;
      if (!framesByDancer.has(key)) framesByDancer.set(key, []);
      framesByDancer.get(key)!.push(f);
    }

    const render = () => {
      const timeMs = video.currentTime * 1000;
      const matchedFrames: FrameData[] = [];
      for (const [, dancerFrames] of framesByDancer) {
        const frame = findFrame(timeMs, dancerFrames);
        if (frame) matchedFrames.push(frame);
      }
      drawSkeletons(matchedFrames, canvas);
      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [perf, findFrame, drawSkeletons]);

  // Resize canvas to match video
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!container || !canvas || !video) return;

    const observer = new ResizeObserver(() => {
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
    });
    observer.observe(video);
    return () => observer.disconnect();
  }, [perf]);

  const handlePlayPause = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
      setPlaying(true);
    } else {
      video.pause();
      setPlaying(false);
    }
  };

  const handleRestart = () => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
  };

  const handleSpeedChange = (s: number) => {
    setSpeed(s);
    if (videoRef.current) videoRef.current.playbackRate = s;
  };

  const handleDelete = async () => {
    if (!performanceId || !confirm("Delete this performance and all analysis data?")) return;
    await deletePerformance(Number(performanceId));
    navigate("/");
  };

  if (loading) {
    return <div className="text-center text-gray-400 py-20">Loading...</div>;
  }

  if (!perf) {
    return <div className="text-center text-gray-400 py-20">Performance not found</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-gray-400 hover:text-white">
            <ArrowLeft size={20} />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-white">
              {perf.item_name || "Performance Review"}
            </h1>
            <p className="text-sm text-gray-400">
              {perf.item_type && <span className="capitalize">{perf.item_type}</span>}
              {perf.talam && <span> | {perf.talam} talam</span>}
              {perf.ragam && <span> | {perf.ragam}</span>}
            </p>
          </div>
        </div>
        <button
          className="flex items-center gap-1 rounded bg-red-800/50 px-3 py-1.5 text-sm text-red-300 hover:bg-red-800"
          onClick={handleDelete}
        >
          <Trash2 size={14} /> Delete
        </button>
      </div>

      {/* Video Player with Skeleton Overlay */}
      <div ref={containerRef} className="relative rounded-lg overflow-hidden bg-black">
        <video
          ref={videoRef}
          src={perf.video_url ?? undefined}
          className="w-full"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between rounded-lg bg-gray-800 p-4">
        <div className="flex items-center gap-3">
          <button
            className="rounded-full bg-brand-600 p-2 text-white hover:bg-brand-700"
            onClick={handlePlayPause}
          >
            {playing ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <button
            className="rounded-full bg-gray-700 p-2 text-gray-300 hover:bg-gray-600"
            onClick={handleRestart}
          >
            <SkipBack size={18} />
          </button>
        </div>

        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-gray-400">
            <input
              type="checkbox"
              checked={showSkeleton}
              onChange={(e) => setShowSkeleton(e.target.checked)}
              className="rounded"
            />
            Skeleton
          </label>

          <div className="flex gap-1">
            {PLAYBACK_SPEEDS.map((s) => (
              <button
                key={s}
                className={`rounded px-2 py-1 text-xs ${
                  speed === s
                    ? "bg-brand-600 text-white"
                    : "bg-gray-700 text-gray-400 hover:bg-gray-600"
                }`}
                onClick={() => handleSpeedChange(s)}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Dancer visibility toggles (multi-dancer mode) */}
      {perf.performance_dancers.length > 1 && (
        <div className="flex gap-2 flex-wrap">
          {perf.performance_dancers.map((pd, idx) => {
            const color = DANCER_COLORS[idx % DANCER_COLORS.length];
            const visible = visibleDancers.has(pd.id);
            return (
              <button
                key={pd.id}
                className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm border transition-all ${
                  visible ? "bg-gray-800 border-gray-600 text-white" : "bg-gray-800/50 border-gray-700 text-gray-500"
                }`}
                onClick={() =>
                  setVisibleDancers((prev) => {
                    const next = new Set(prev);
                    if (next.has(pd.id)) next.delete(pd.id);
                    else next.add(pd.id);
                    return next;
                  })
                }
              >
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color, opacity: visible ? 1 : 0.3 }} />
                {pd.label || `Dancer ${idx + 1}`}
              </button>
            );
          })}
        </div>
      )}

      {/* Coaching Feedback */}
      {perf.performance_dancers.length > 1 ? (
        <section className="rounded-lg bg-gray-800 p-6 space-y-4">
          <h2 className="text-lg font-semibold text-brand-400">Coaching Feedback</h2>
          {/* Dancer tabs */}
          <div className="flex gap-2 border-b border-gray-700 pb-2">
            {perf.performance_dancers.map((pd, idx) => {
              const color = DANCER_COLORS[idx % DANCER_COLORS.length];
              return (
                <button
                  key={pd.id}
                  className={`flex items-center gap-2 rounded-t-lg px-4 py-2 text-sm transition-all ${
                    activeDancerTab === pd.id
                      ? "bg-gray-700 text-white"
                      : "text-gray-400 hover:text-white"
                  }`}
                  onClick={() => setActiveDancerTab(pd.id)}
                >
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                  {pd.label || `Dancer ${idx + 1}`}
                </button>
              );
            })}
          </div>
          {/* Active dancer analysis */}
          {(() => {
            const analysis = perf.analysis.find((a) => a.performance_dancer_id === activeDancerTab);
            if (!analysis) return <p className="text-gray-500">No analysis available for this dancer.</p>;
            return (
              <>
                <ScoreCards analysis={analysis} />
                {analysis.llm_summary && (
                  <div className="prose prose-invert max-w-none text-sm text-gray-300 whitespace-pre-wrap leading-relaxed">
                    {analysis.llm_summary}
                  </div>
                )}
              </>
            );
          })()}
        </section>
      ) : (
        (() => {
          const analysis = perf.analysis?.[0];
          return analysis ? (
            <section className="rounded-lg bg-gray-800 p-6 space-y-4">
              <h2 className="text-lg font-semibold text-brand-400">Coaching Feedback</h2>
              <ScoreCards analysis={analysis} />
              {analysis.llm_summary && (
                <div className="prose prose-invert max-w-none text-sm text-gray-300 whitespace-pre-wrap leading-relaxed">
                  {analysis.llm_summary}
                </div>
              )}
            </section>
          ) : null;
        })()
      )}

      {/* Frame count info */}
      <div className="text-center text-xs text-gray-500">
        {framesLoading ? "Loading skeleton data..." : `${frames.length} frames analyzed`}
        {perf.duration_ms && ` | ${(perf.duration_ms / 1000).toFixed(1)}s duration`}
      </div>
    </div>
  );
}
