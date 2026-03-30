import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, Users } from "lucide-react";
import { getPerformance, selectFrame, Performance, ClickPrompt } from "../api/performances";

const DANCER_COLORS = [
  "#F9A825", "#00BCD4", "#E91E63", "#8BC34A",
  "#FF7043", "#7C4DFF", "#26C6DA", "#FFD54F",
];

interface DancerSelection {
  x: number;
  y: number;
  label: string;
  color: string;
}

export default function SelectFrame() {
  const { performanceId } = useParams<{ performanceId: string }>();
  const navigate = useNavigate();
  const [perf, setPerf] = useState<Performance | null>(null);
  const [dancers, setDancers] = useState<DancerSelection[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!performanceId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const load = async () => {
      try {
        const data = await getPerformance(Number(performanceId));
        if (cancelled) return;
        setPerf(data);
        // Keep polling until transcode finishes
        if (data.status !== "uploaded") {
          timer = setTimeout(load, 1500);
        }
      } catch {
        if (!cancelled) navigate("/");
      }
    };
    load();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [performanceId, navigate]);

  // Sync canvas size to video display size
  const syncCanvasSize = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
  }, []);

  // Draw dots on canvas
  const drawDots = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const d of dancers) {
      const px = d.x * canvas.width;
      const py = d.y * canvas.height;
      ctx.beginPath();
      ctx.arc(px, py, 12, 0, Math.PI * 2);
      ctx.fillStyle = d.color;
      ctx.fill();
      ctx.strokeStyle = "white";
      ctx.lineWidth = 2;
      ctx.stroke();
      // Label
      ctx.fillStyle = "white";
      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(d.label || `Dancer ${dancers.indexOf(d) + 1}`, px, py - 18);
    }
  }, [dancers]);

  useEffect(() => { drawDots(); }, [drawDots]);

  // Redraw on window resize
  useEffect(() => {
    const handleResize = () => { syncCanvasSize(); drawDots(); };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [syncCanvasSize, drawDots]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    // Pause video on click
    if (videoRef.current) videoRef.current.pause();

    setDancers(prev => [
      ...prev,
      { x, y, label: "", color: DANCER_COLORS[prev.length % DANCER_COLORS.length] },
    ]);
  };

  const handleRemove = (index: number) => {
    setDancers(prev => prev.filter((_, i) => i !== index));
  };

  const handleLabelChange = (index: number, label: string) => {
    setDancers(prev => prev.map((d, i) => i === index ? { ...d, label } : d));
  };

  const handleSubmit = async () => {
    if (!performanceId || dancers.length === 0 || !videoRef.current) return;
    setSubmitting(true);
    setError("");
    const startMs = Math.round(videoRef.current.currentTime * 1000);
    const prompts: ClickPrompt[] = dancers.map((d, i) => ({
      x: d.x,
      y: d.y,
      label: d.label || `Dancer ${i + 1}`,
    }));
    try {
      await selectFrame(Number(performanceId), startMs, prompts);
      navigate(`/processing/${performanceId}`);
    } catch {
      setError("Failed to start tracking. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const isReady = perf?.status === "uploaded";

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <button onClick={() => navigate("/")} className="text-gray-400 hover:text-white">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-white">Select Dancers</h1>
          <p className="text-gray-400 text-sm">
            Scrub to a clear frame, then click on each dancer you want to track
          </p>
        </div>
      </div>

      {!isReady && perf?.status === "transcoding" && (
        <div className="rounded-lg bg-gray-800 p-8 text-center text-gray-400">
          <div className="animate-spin inline-block w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full mb-4" />
          <p>Preparing video for playback...</p>
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-red-900/30 border border-red-700 p-4 text-red-300">{error}</div>
      )}

      {isReady && perf?.video_url && (
        <>
          {/* Video with canvas overlay */}
          <div className="relative rounded-lg overflow-hidden bg-black">
            <video
              ref={videoRef}
              src={perf.video_url}
              controls
              className="w-full"
              onLoadedMetadata={drawDots}
              onSeeked={drawDots}
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 cursor-crosshair"
              onClick={handleCanvasClick}
              style={{ pointerEvents: "auto" }}
            />
          </div>

          {/* Selected dancers list */}
          {dancers.length > 0 && (
            <div className="rounded-lg bg-gray-800 p-4 space-y-3">
              <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                <Users size={16} /> Selected Dancers ({dancers.length})
              </h3>
              {dancers.map((d, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-4 h-4 rounded-full" style={{ backgroundColor: d.color }} />
                  <input
                    className="flex-1 rounded bg-gray-700 px-3 py-1.5 text-sm text-white"
                    placeholder={`Dancer ${i + 1}`}
                    value={d.label}
                    onChange={(e) => handleLabelChange(i, e.target.value)}
                  />
                  <button
                    className="text-red-400 hover:text-red-300 text-sm"
                    onClick={() => handleRemove(i)}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Submit button */}
          <button
            className="w-full rounded-lg bg-brand-600 py-3 text-lg font-semibold text-white hover:bg-brand-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            disabled={dancers.length === 0 || submitting}
            onClick={handleSubmit}
          >
            {submitting ? "Starting analysis..." : `Analyze from ${videoRef.current ? formatTime(videoRef.current.currentTime) : "here"}`}
          </button>
        </>
      )}
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
