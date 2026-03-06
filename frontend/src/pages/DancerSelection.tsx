import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Users, Check, ArrowLeft } from "lucide-react";
import {
  getPerformance,
  getDetectedPersons,
  selectDancers,
  DetectedPerson,
  Performance,
} from "../api/performances";

const DANCER_COLORS = [
  "#F9A825", // saffron
  "#00BCD4", // teal
  "#E91E63", // magenta
  "#8BC34A", // lime
  "#FF7043", // coral
  "#7C4DFF", // violet
  "#26C6DA", // cyan
  "#FFD54F", // gold
];

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
  ["nose", "left_shoulder"],
  ["nose", "right_shoulder"],
];

interface DancerState {
  trackId: number;
  selected: boolean;
  label: string;
  color: string;
}

export default function DancerSelection() {
  const { performanceId } = useParams<{ performanceId: string }>();
  const navigate = useNavigate();
  const [perf, setPerf] = useState<Performance | null>(null);
  const [persons, setPersons] = useState<DetectedPerson[]>([]);
  const [dancers, setDancers] = useState<DancerState[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!performanceId) return;
    setLoading(true);
    Promise.all([
      getPerformance(Number(performanceId)),
      getDetectedPersons(Number(performanceId)),
    ])
      .then(([perfData, personsData]) => {
        setPerf(perfData);
        setPersons(personsData);
        setDancers(
          personsData.map((p, i) => ({
            trackId: p.track_id,
            selected: true,
            label: personsData.length === 1 ? "Dancer" : `Dancer ${i + 1}`,
            color: DANCER_COLORS[i % DANCER_COLORS.length],
          }))
        );
      })
      .catch(() => navigate("/"))
      .finally(() => setLoading(false));
  }, [performanceId, navigate]);

  const drawSkeletons = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !persons.length) return;

    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const w = canvas.width;
    const h = canvas.height;

    persons.forEach((person, idx) => {
      const state = dancers[idx];
      if (!state) return;

      const color = state.color;
      const alpha = state.selected ? 1.0 : 0.2;
      const pose = person.representative_pose;
      if (!pose) return;

      // Draw bounding box
      const bbox = person.bbox;
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(
        bbox.x_min * w,
        bbox.y_min * h,
        (bbox.x_max - bbox.x_min) * w,
        (bbox.y_max - bbox.y_min) * h
      );
      ctx.setLineDash([]);

      // Draw label tag
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha * 0.85;
      const labelText = state.label || `Person ${idx + 1}`;
      ctx.font = "bold 13px sans-serif";
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillRect(bbox.x_min * w, bbox.y_min * h - 20, textWidth + 12, 20);
      ctx.fillStyle = "#000";
      ctx.globalAlpha = alpha;
      ctx.fillText(labelText, bbox.x_min * w + 6, bbox.y_min * h - 5);

      // Draw skeleton
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      for (const [from, to] of SKELETON_CONNECTIONS) {
        const a = pose[from];
        const b = pose[to];
        if (!a || !b || a.confidence < 0.3 || b.confidence < 0.3) continue;
        ctx.beginPath();
        ctx.moveTo(a.x * w, a.y * h);
        ctx.lineTo(b.x * w, b.y * h);
        ctx.stroke();
      }

      // Draw keypoints
      for (const kp of Object.values(pose) as { x: number; y: number; confidence: number }[]) {
        if (kp.confidence < 0.3) continue;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(kp.x * w, kp.y * h, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    ctx.globalAlpha = 1.0;
  }, [persons, dancers]);

  useEffect(() => {
    drawSkeletons();
  }, [drawSkeletons]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const observer = new ResizeObserver(drawSkeletons);
    observer.observe(img);
    return () => observer.disconnect();
  }, [drawSkeletons]);

  const toggleDancer = (idx: number) => {
    setDancers((prev) =>
      prev.map((d, i) => (i === idx ? { ...d, selected: !d.selected } : d))
    );
  };

  const updateLabel = (idx: number, label: string) => {
    setDancers((prev) =>
      prev.map((d, i) => (i === idx ? { ...d, label } : d))
    );
  };

  const handleSubmit = async () => {
    if (!performanceId) return;
    const selected = dancers.filter((d) => d.selected);
    if (selected.length === 0) return;

    setSubmitting(true);
    try {
      await selectDancers(
        Number(performanceId),
        selected.map((d) => ({ track_id: d.trackId, label: d.label || undefined }))
      );
      navigate(`/processing/${performanceId}`);
    } catch {
      setSubmitting(false);
    }
  };

  if (loading) {
    return <div className="text-center text-gray-400 py-20">Loading detected persons...</div>;
  }

  if (!perf || persons.length === 0) {
    return (
      <div className="text-center text-gray-400 py-20">
        <p>No persons detected in the video.</p>
        <button
          className="mt-4 rounded-lg bg-gray-700 px-6 py-2 text-white hover:bg-gray-600"
          onClick={() => navigate("/")}
        >
          Back to Dashboard
        </button>
      </div>
    );
  }

  const selectedCount = dancers.filter((d) => d.selected).length;
  const apiBase = import.meta.env.VITE_API_URL || "http://localhost:8000";

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button onClick={() => navigate("/")} className="text-gray-400 hover:text-white">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Users size={24} className="text-brand-400" />
            Select Dancers to Analyze
          </h1>
          <p className="mt-1 text-gray-400">
            {persons.length} person{persons.length !== 1 ? "s" : ""} detected.
            Select who to track and optionally name them.
          </p>
        </div>
      </div>

      {/* Detection frame with skeleton overlays */}
      <div className="relative rounded-lg overflow-hidden bg-black">
        {perf.detection_frame_url && (
          <img
            ref={imgRef}
            src={`${apiBase}${perf.detection_frame_url}`}
            alt="Detection frame"
            className="w-full"
            onLoad={drawSkeletons}
          />
        )}
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* Dancer cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {dancers.map((dancer, idx) => (
          <div
            key={dancer.trackId}
            className={`rounded-lg border-2 p-4 transition-all cursor-pointer ${
              dancer.selected
                ? "border-brand-500 bg-gray-800"
                : "border-gray-700 bg-gray-800/50 opacity-60"
            }`}
            onClick={() => toggleDancer(idx)}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: dancer.color }}
                />
                <span className="text-white font-medium">Person {idx + 1}</span>
              </div>
              <div
                className={`w-6 h-6 rounded-md flex items-center justify-center ${
                  dancer.selected ? "bg-brand-500" : "bg-gray-700"
                }`}
              >
                {dancer.selected && <Check size={14} className="text-white" />}
              </div>
            </div>
            <input
              type="text"
              value={dancer.label}
              onChange={(e) => {
                e.stopPropagation();
                updateLabel(idx, e.target.value);
              }}
              onClick={(e) => e.stopPropagation()}
              placeholder="Name this dancer..."
              className="w-full rounded bg-gray-700 px-3 py-1.5 text-sm text-white placeholder-gray-500 border border-gray-600 focus:border-brand-500 focus:outline-none"
            />
            <div className="mt-2 text-xs text-gray-500">
              Visible in {persons[idx].frame_count} frames
            </div>
          </div>
        ))}
      </div>

      {/* Submit */}
      <div className="flex justify-center">
        <button
          className="rounded-lg bg-brand-600 px-8 py-3 text-white font-medium hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={selectedCount === 0 || submitting}
          onClick={handleSubmit}
        >
          {submitting
            ? "Starting analysis..."
            : `Analyze ${selectedCount} Dancer${selectedCount !== 1 ? "s" : ""}`}
        </button>
      </div>
    </div>
  );
}
