import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { Play, Pause, SkipBack, Trash2, ArrowLeft } from "lucide-react";
import { getPerformance, getPerformanceFrames, getPerformanceTimeline, deletePerformance, Performance, FrameData, AnalysisData, TimelineFrame } from "../api/performances";

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

interface ScoreDetail {
  label: string;
  value: number | null;
  measured?: string;
  ideal?: string;
  tip?: string;
}

function getScoreDetails(analysis: AnalysisData): ScoreDetail[] {
  const inputs = analysis.technique_scores?.inputs as Record<string, number> | undefined;

  const fmt = (v: number | undefined | null, unit: string, decimals = 1) =>
    v != null ? `${v.toFixed(decimals)}${unit}` : "N/A";

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
    {
      label: "Rhythm",
      value: analysis.rhythm_consistency_score != null ? Math.round(analysis.rhythm_consistency_score) : null,
      measured: inputs?.rhythm_score !== undefined
        ? `Match rate: ${fmt((analysis.technique_scores?.rhythm_details as Record<string, number> | undefined)?.match_rate, "%", 0)} | Avg offset: ${fmt((analysis.technique_scores?.rhythm_details as Record<string, number> | undefined)?.avg_offset_ms, "ms", 0)}${inputs.tempo_bpm ? ` | Tempo: ${fmt(inputs.tempo_bpm, " BPM", 0)}` : ""}`
        : undefined,
      ideal: "100% foot strikes aligned with musical onsets within 75ms",
      tip: analysis.rhythm_consistency_score != null
        ? analysis.rhythm_consistency_score < 30
          ? "Your foot strikes are rarely aligned with the music. Practice with a metronome or nattuvangam recording, focusing on landing stamps on each beat."
          : analysis.rhythm_consistency_score < 60
          ? "Some strikes land on beat, but timing is inconsistent. Slow down and practice each jathi phrase until stamps consistently match the syllables."
          : analysis.rhythm_consistency_score < 80
          ? "Good rhythmic sense. Fine-tune by listening closely to the mridangam strokes and aligning your thattu (stamps) precisely."
          : "Excellent rhythm — your foot strikes closely match the musical onsets."
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
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
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

// --- Movement classification from per-frame angles ---

type MoveType = "aramandi" | "standing" | "arms_extended" | "transition" | "unknown";

const MOVE_COLORS: Record<MoveType, string> = {
  aramandi: "#F9A825",      // saffron
  standing: "#4CAF50",      // green
  arms_extended: "#2196F3", // blue
  transition: "#9E9E9E",    // gray
  unknown: "#424242",       // dark gray
};

const MOVE_LABELS: Record<MoveType, string> = {
  aramandi: "Aramandi",
  standing: "Standing",
  arms_extended: "Arms Extended",
  transition: "Transition",
  unknown: "Unknown",
};

function classifyMove(f: TimelineFrame): MoveType {
  if (f.aramandi_angle == null) return "unknown";
  const knee = f.aramandi_angle;
  const armL = f.arm_extension_left ?? 0;
  const armR = f.arm_extension_right ?? 0;
  const avgArm = (armL + armR) / 2;

  if (knee <= 130) return "aramandi";
  if (avgArm > 150) return "arms_extended";
  if (knee > 160) return "standing";
  return "transition";
}

interface MoveSegment {
  move: MoveType;
  startMs: number;
  endMs: number;
}

function buildSegments(frames: TimelineFrame[]): MoveSegment[] {
  if (!frames.length) return [];
  const segments: MoveSegment[] = [];
  let current: MoveSegment = { move: classifyMove(frames[0]), startMs: frames[0].timestamp_ms, endMs: frames[0].timestamp_ms };

  for (let i = 1; i < frames.length; i++) {
    const move = classifyMove(frames[i]);
    if (move === current.move) {
      current.endMs = frames[i].timestamp_ms;
    } else {
      segments.push(current);
      current = { move, startMs: frames[i].timestamp_ms, endMs: frames[i].timestamp_ms };
    }
  }
  segments.push(current);

  // Merge very short segments (< 200ms) into neighbors
  const merged: MoveSegment[] = [];
  for (const seg of segments) {
    if (merged.length > 0 && seg.endMs - seg.startMs < 200) {
      merged[merged.length - 1].endMs = seg.endMs;
    } else {
      merged.push({ ...seg });
    }
  }
  return merged;
}

function computeSyncScore(framesA: TimelineFrame[], framesB: TimelineFrame[]): { syncPercent: number; perSecond: { timeSec: number; sync: number }[] } {
  if (!framesA.length || !framesB.length) return { syncPercent: 0, perSecond: [] };

  // Build a map of timestamp -> move for dancer B
  const bMoves = new Map<number, MoveType>();
  for (const f of framesB) {
    bMoves.set(f.timestamp_ms, classifyMove(f));
  }

  let matchCount = 0;
  const buckets = new Map<number, { match: number; total: number }>();

  for (const f of framesA) {
    const moveA = classifyMove(f);
    // Find closest B frame
    const moveB = bMoves.get(f.timestamp_ms);
    const timeSec = Math.floor(f.timestamp_ms / 1000);
    if (!buckets.has(timeSec)) buckets.set(timeSec, { match: 0, total: 0 });
    const bucket = buckets.get(timeSec)!;
    bucket.total++;
    if (moveB && moveA === moveB) {
      matchCount++;
      bucket.match++;
    }
  }

  const syncPercent = Math.round((matchCount / framesA.length) * 100);
  const perSecond = Array.from(buckets.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([timeSec, { match, total }]) => ({ timeSec, sync: total > 0 ? match / total : 0 }));

  return { syncPercent, perSecond };
}

interface DanceTimelineProps {
  timelineData: TimelineFrame[];
  dancers: { id: number; label: string | null }[];
  durationMs: number;
  currentTimeMs: number;
  beatTimestamps: number[] | null;
  tempoBpm: number | null;
  onSeek: (timeMs: number) => void;
}

function DanceTimeline({ timelineData, dancers, durationMs, currentTimeMs, beatTimestamps, tempoBpm, onSeek }: DanceTimelineProps) {
  const { dancerIds, segmentsByDancer, syncInfo, usedMoves } = useMemo(() => {
    if (!timelineData.length) {
      return { dancerIds: [] as (number | null)[], segmentsByDancer: new Map<number | null, MoveSegment[]>(), syncInfo: null, usedMoves: new Set<MoveType>() };
    }
    // Group by dancer
    const byDancer = new Map<number | null, TimelineFrame[]>();
    for (const f of timelineData) {
      const key = f.performance_dancer_id;
      if (!byDancer.has(key)) byDancer.set(key, []);
      byDancer.get(key)!.push(f);
    }

    const ids = dancers.length > 0 ? dancers.map((d) => d.id) : [null as number | null];
    const segments = new Map<number | null, MoveSegment[]>();
    for (const did of ids) {
      const frames = byDancer.get(did) ?? [];
      segments.set(did, buildSegments(frames));
    }

    // Compute sync between first two dancers if multi-dancer
    let sync: { syncPercent: number; perSecond: { timeSec: number; sync: number }[] } | null = null;
    if (ids.length >= 2 && ids[0] != null && ids[1] != null) {
      sync = computeSyncScore(byDancer.get(ids[0]) ?? [], byDancer.get(ids[1]) ?? []);
    }

    // Collect unique moves present
    const moves = new Set<MoveType>();
    for (const [, segs] of segments) {
      for (const seg of segs) moves.add(seg.move);
    }

    return { dancerIds: ids, segmentsByDancer: segments, syncInfo: sync, usedMoves: moves };
  }, [timelineData, dancers]);

  if (!timelineData.length || !durationMs) return null;

  const playheadPct = durationMs > 0 ? (currentTimeMs / durationMs) * 100 : 0;

  return (
    <div className="rounded-lg bg-gray-800 p-4 space-y-3">
      <h3 className="text-sm font-semibold text-gray-300">Movement Timeline</h3>

      {dancerIds.map((did, dIdx) => {
        const segments = segmentsByDancer.get(did) ?? [];
        const dancer = dancers.find((d) => d.id === did);
        const color = did != null ? DANCER_COLORS[dIdx % DANCER_COLORS.length] : SAFFRON;

        return (
          <div key={did ?? "solo"} className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-xs text-gray-400">{dancer?.label || `Dancer ${dIdx + 1}`}</span>
            </div>
            <div
              className="relative h-6 bg-gray-700 rounded cursor-pointer overflow-hidden"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const pct = (e.clientX - rect.left) / rect.width;
                onSeek(pct * durationMs);
              }}
            >
              {segments.map((seg, i) => {
                const left = (seg.startMs / durationMs) * 100;
                const width = Math.max(0.3, ((seg.endMs - seg.startMs) / durationMs) * 100);
                return (
                  <div
                    key={i}
                    className="absolute top-0 h-full opacity-80 hover:opacity-100 transition-opacity"
                    style={{ left: `${left}%`, width: `${width}%`, backgroundColor: MOVE_COLORS[seg.move] }}
                    title={`${MOVE_LABELS[seg.move]} (${((seg.endMs - seg.startMs) / 1000).toFixed(1)}s)`}
                  />
                );
              })}
              {/* Playhead */}
              <div className="absolute top-0 h-full w-0.5 bg-white z-10" style={{ left: `${playheadPct}%` }} />
            </div>
          </div>
        );
      })}

      {/* Synchronicity bar */}
      {syncInfo && (
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">Synchronicity</span>
            <span className="text-xs font-medium text-brand-400">{syncInfo.syncPercent}%</span>
          </div>
          <div
            className="relative h-3 bg-gray-700 rounded cursor-pointer overflow-hidden"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const pct = (e.clientX - rect.left) / rect.width;
              onSeek(pct * durationMs);
            }}
          >
            {syncInfo.perSecond.map((ps) => {
              const left = (ps.timeSec * 1000 / durationMs) * 100;
              const width = Math.max(0.2, (1000 / durationMs) * 100);
              const green = Math.round(ps.sync * 200 + 55);
              const red = Math.round((1 - ps.sync) * 200 + 55);
              return (
                <div
                  key={ps.timeSec}
                  className="absolute top-0 h-full"
                  style={{
                    left: `${left}%`,
                    width: `${width}%`,
                    backgroundColor: `rgb(${red}, ${green}, 80)`,
                  }}
                  title={`${ps.timeSec}s: ${Math.round(ps.sync * 100)}% sync`}
                />
              );
            })}
            <div className="absolute top-0 h-full w-0.5 bg-white z-10" style={{ left: `${playheadPct}%` }} />
          </div>
        </div>
      )}

      {/* Beat markers */}
      {beatTimestamps && beatTimestamps.length > 0 && (
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">Beat / Onset Markers</span>
            {tempoBpm && <span className="text-xs font-medium text-brand-400">{tempoBpm} BPM</span>}
            <span className="text-xs text-gray-500">{beatTimestamps.length} onsets</span>
          </div>
          <div
            className="relative h-4 bg-gray-700 rounded cursor-pointer overflow-hidden"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const pct = (e.clientX - rect.left) / rect.width;
              onSeek(pct * durationMs);
            }}
          >
            {beatTimestamps.map((ts, i) => {
              const left = (ts / durationMs) * 100;
              return (
                <div
                  key={i}
                  className="absolute top-0 w-px h-full bg-yellow-400 opacity-50 hover:opacity-100"
                  style={{ left: `${left}%` }}
                  title={`Onset ${i + 1}: ${(ts / 1000).toFixed(2)}s`}
                />
              );
            })}
            <div className="absolute top-0 h-full w-0.5 bg-white z-10" style={{ left: `${playheadPct}%` }} />
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="flex flex-wrap gap-3 pt-1">
        {Array.from(usedMoves).map((move) => (
          <div key={move} className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: MOVE_COLORS[move] }} />
            <span className="text-xs text-gray-400">{MOVE_LABELS[move]}</span>
          </div>
        ))}
      </div>
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

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [timelineData, setTimelineData] = useState<TimelineFrame[]>([]);
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

    getPerformanceTimeline(id)
      .then(setTimelineData)
      .catch(() => {});
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
    [visibleDancers, getDancerColor]
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
  }, [perf, frames, findFrame, drawSkeletons]);

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

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
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
          onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime ?? 0)}
          onLoadedMetadata={() => setDuration(videoRef.current?.duration ?? 0)}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* Controls */}
      <div className="flex flex-col gap-3 rounded-lg bg-gray-800 p-4">
        <div className="flex items-center gap-3">
          <button
            className="rounded-full bg-brand-600 p-2 text-white hover:bg-brand-700 shrink-0"
            onClick={handlePlayPause}
          >
            {playing ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <button
            className="rounded-full bg-gray-700 p-2 text-gray-300 hover:bg-gray-600 shrink-0"
            onClick={handleRestart}
          >
            <SkipBack size={18} />
          </button>

          {/* Seekbar */}
          <span className="text-xs text-gray-400 tabular-nums shrink-0">
            {formatTime(currentTime)}
          </span>
          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.01}
            value={currentTime}
            onChange={handleSeek}
            className="flex-1 h-1.5 cursor-pointer appearance-none rounded-full bg-gray-600
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3.5
              [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-brand-400 [&::-webkit-slider-thumb]:hover:bg-brand-300
              [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5
              [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-brand-400
              [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:hover:bg-brand-300
              [&::-moz-range-track]:bg-gray-600 [&::-moz-range-track]:rounded-full"
          />
          <span className="text-xs text-gray-400 tabular-nums shrink-0">
            {formatTime(duration)}
          </span>
        </div>

        <div className="flex items-center justify-end gap-4">
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

      {/* Dancer skeleton visibility toggles */}
      {perf.performance_dancers.length > 0 && (
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

      {/* Dance Move Timeline */}
      <DanceTimeline
        timelineData={timelineData}
        dancers={perf.performance_dancers}
        durationMs={perf.duration_ms ?? duration * 1000}
        currentTimeMs={currentTime * 1000}
        beatTimestamps={perf.beat_timestamps}
        tempoBpm={perf.tempo_bpm}
        onSeek={(ms) => {
          if (videoRef.current) {
            videoRef.current.currentTime = ms / 1000;
            setCurrentTime(ms / 1000);
          }
        }}
      />

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
