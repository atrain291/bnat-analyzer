import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Users, Link2, ArrowLeft, Camera } from "lucide-react";
import {
  getMultiAngleGroup,
  linkDancersAcrossViews,
  MultiAngleGroup,
  CrossViewDancerLink,
} from "../api/multiAngle";
import { getDetectedPersons, DetectedPerson } from "../api/performances";

const DANCER_COLORS = [
  "#F9A825", "#00BCD4", "#E91E63", "#8BC34A",
  "#FF7043", "#7C4DFF", "#26C6DA", "#FFD54F",
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

interface PerViewData {
  performanceId: number;
  cameraLabel: string;
  detectionFrameUrl: string | null;
  persons: DetectedPerson[];
  // For each person, which linked dancer group they're assigned to (index into linkedDancers)
  assignments: (number | null)[];
}

interface LinkedDancer {
  label: string;
  color: string;
}

export default function MultiAngleDancerLink() {
  const { groupId } = useParams<{ groupId: string }>();
  const navigate = useNavigate();
  const [, setGroup] = useState<MultiAngleGroup | null>(null);
  const [perViewData, setPerViewData] = useState<PerViewData[]>([]);
  const [linkedDancers, setLinkedDancers] = useState<LinkedDancer[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const imgRefs = useRef<(HTMLImageElement | null)[]>([]);

  useEffect(() => {
    if (!groupId) return;
    setLoading(true);

    getMultiAngleGroup(Number(groupId))
      .then(async (groupData) => {
        setGroup(groupData);

        // Fetch detected persons for each performance
        const viewData: PerViewData[] = [];
        for (const perf of groupData.performances) {
          const persons = await getDetectedPersons(perf.id);
          viewData.push({
            performanceId: perf.id,
            cameraLabel: perf.camera_label || `Camera`,
            detectionFrameUrl: perf.detection_frame_url,
            persons,
            assignments: persons.map(() => null),
          });
        }
        setPerViewData(viewData);

        // Auto-create one linked dancer if each view has exactly one person
        const allSingle = viewData.every((v) => v.persons.length === 1);
        if (allSingle) {
          setLinkedDancers([{ label: "Dancer", color: DANCER_COLORS[0] }]);
          setPerViewData(viewData.map((v) => ({ ...v, assignments: [0] })));
        }
      })
      .catch(() => navigate("/"))
      .finally(() => setLoading(false));
  }, [groupId, navigate]);

  const drawSkeletons = useCallback(
    (viewIdx: number) => {
      const canvas = canvasRefs.current[viewIdx];
      const img = imgRefs.current[viewIdx];
      const view = perViewData[viewIdx];
      if (!canvas || !img || !view) return;

      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const w = canvas.width;
      const h = canvas.height;

      view.persons.forEach((person, pIdx) => {
        const assignedIdx = view.assignments[pIdx];
        const linked = assignedIdx !== null ? linkedDancers[assignedIdx] : null;
        const color = linked ? linked.color : "#888";
        const alpha = linked ? 1.0 : 0.4;

        const pose = person.representative_pose;
        if (!pose) return;

        // Bounding box
        const bbox = person.bbox;
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(bbox.x_min * w, bbox.y_min * h, (bbox.x_max - bbox.x_min) * w, (bbox.y_max - bbox.y_min) * h);
        ctx.setLineDash([]);

        // Label
        const labelText = linked ? linked.label : `Person ${pIdx + 1}`;
        ctx.fillStyle = color;
        ctx.globalAlpha = alpha * 0.85;
        ctx.font = "bold 12px sans-serif";
        const tw = ctx.measureText(labelText).width;
        ctx.fillRect(bbox.x_min * w, bbox.y_min * h - 18, tw + 10, 18);
        ctx.fillStyle = "#000";
        ctx.globalAlpha = alpha;
        ctx.fillText(labelText, bbox.x_min * w + 5, bbox.y_min * h - 4);

        // Skeleton
        ctx.lineWidth = 2;
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

        // Keypoints
        for (const kp of Object.values(pose) as { x: number; y: number; confidence: number }[]) {
          if (kp.confidence < 0.3) continue;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(kp.x * w, kp.y * h, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      });
      ctx.globalAlpha = 1.0;
    },
    [perViewData, linkedDancers],
  );

  useEffect(() => {
    perViewData.forEach((_, i) => drawSkeletons(i));
  }, [drawSkeletons, perViewData]);

  const addLinkedDancer = () => {
    const idx = linkedDancers.length;
    setLinkedDancers((prev) => [
      ...prev,
      { label: `Dancer ${idx + 1}`, color: DANCER_COLORS[idx % DANCER_COLORS.length] },
    ]);
  };

  const updateLinkedDancerLabel = (idx: number, label: string) => {
    setLinkedDancers((prev) => prev.map((d, i) => (i === idx ? { ...d, label } : d)));
  };

  const assignPerson = (viewIdx: number, personIdx: number, dancerIdx: number | null) => {
    setPerViewData((prev) =>
      prev.map((v, vi) => {
        if (vi !== viewIdx) return v;
        const newAssignments = [...v.assignments];
        newAssignments[personIdx] = dancerIdx;
        return { ...v, assignments: newAssignments };
      }),
    );
  };

  const handleSubmit = async () => {
    if (!groupId) return;

    // Build cross-view links
    const links: CrossViewDancerLink[] = [];
    for (let di = 0; di < linkedDancers.length; di++) {
      const performanceTracks: Record<number, number> = {};
      for (const view of perViewData) {
        const personIdx = view.assignments.indexOf(di);
        if (personIdx >= 0) {
          performanceTracks[view.performanceId] = view.persons[personIdx].track_id;
        }
      }
      if (Object.keys(performanceTracks).length > 0) {
        links.push({ label: linkedDancers[di].label, performance_tracks: performanceTracks });
      }
    }

    if (links.length === 0) return;

    setSubmitting(true);
    try {
      await linkDancersAcrossViews(Number(groupId), links);
      navigate(`/multi-angle/processing/${groupId}`);
    } catch {
      setSubmitting(false);
    }
  };

  if (loading) {
    return <div className="text-center text-gray-400 py-20">Loading detection results...</div>;
  }

  const anyAssigned = perViewData.some((v) => v.assignments.some((a) => a !== null));
  const apiBase = import.meta.env.VITE_API_URL || "http://localhost:8000";

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="flex items-center gap-4">
        <button onClick={() => navigate("/")} className="text-gray-400 hover:text-white">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Link2 size={24} className="text-brand-400" />
            Link Dancers Across Views
          </h1>
          <p className="mt-1 text-gray-400">
            Match each dancer across camera angles so scores can be combined
          </p>
        </div>
      </div>

      {/* Linked dancer groups */}
      <section className="rounded-lg bg-gray-800 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-white font-medium flex items-center gap-2">
            <Users size={18} /> Dancers to Track
          </h2>
          <button
            className="text-sm text-brand-400 hover:text-brand-300 flex items-center gap-1"
            onClick={addLinkedDancer}
          >
            + Add Dancer
          </button>
        </div>
        {linkedDancers.map((d, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="w-4 h-4 rounded-full flex-shrink-0" style={{ backgroundColor: d.color }} />
            <input
              type="text"
              value={d.label}
              onChange={(e) => updateLinkedDancerLabel(i, e.target.value)}
              className="flex-1 rounded bg-gray-700 px-3 py-1.5 text-sm text-white border border-gray-600 focus:border-brand-500 focus:outline-none"
              placeholder="Dancer name..."
            />
          </div>
        ))}
        {linkedDancers.length === 0 && (
          <p className="text-gray-500 text-sm">Click "+ Add Dancer" to create a dancer, then assign them in each view below</p>
        )}
      </section>

      {/* Per-view detection frames with assignment dropdowns */}
      <div className={`grid gap-6 ${perViewData.length === 2 ? "grid-cols-2" : "grid-cols-1 lg:grid-cols-2"}`}>
        {perViewData.map((view, viewIdx) => (
          <section key={view.performanceId} className="rounded-lg bg-gray-800 p-4 space-y-3">
            <h3 className="text-white font-medium flex items-center gap-2">
              <Camera size={16} className="text-brand-400" />
              {view.cameraLabel}
            </h3>

            {/* Detection frame */}
            <div className="relative rounded overflow-hidden bg-black">
              {view.detectionFrameUrl && (
                <img
                  ref={(el) => { imgRefs.current[viewIdx] = el; }}
                  src={`${apiBase}${view.detectionFrameUrl}`}
                  alt={`${view.cameraLabel} detection`}
                  className="w-full"
                  onLoad={() => drawSkeletons(viewIdx)}
                />
              )}
              <canvas
                ref={(el) => { canvasRefs.current[viewIdx] = el; }}
                className="absolute top-0 left-0 w-full h-full pointer-events-none"
              />
            </div>

            {/* Person assignment */}
            <div className="space-y-2">
              {view.persons.map((person, pIdx) => (
                <div key={person.id} className="flex items-center gap-2 text-sm">
                  <span className="text-gray-400 w-20">Person {pIdx + 1}</span>
                  <select
                    className="flex-1 rounded bg-gray-700 px-2 py-1 text-white text-sm"
                    value={view.assignments[pIdx] ?? ""}
                    onChange={(e) => {
                      const val = e.target.value;
                      assignPerson(viewIdx, pIdx, val === "" ? null : Number(val));
                    }}
                  >
                    <option value="">Not tracked</option>
                    {linkedDancers.map((d, di) => (
                      <option key={di} value={di}>{d.label}</option>
                    ))}
                  </select>
                  {view.assignments[pIdx] !== null && (
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: linkedDancers[view.assignments[pIdx]!]?.color }}
                    />
                  )}
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>

      {/* Submit */}
      <div className="flex justify-center">
        <button
          className="rounded-lg bg-brand-600 px-8 py-3 text-white font-medium hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={!anyAssigned || linkedDancers.length === 0 || submitting}
          onClick={handleSubmit}
        >
          {submitting ? "Starting analysis..." : "Analyze All Angles"}
        </button>
      </div>
    </div>
  );
}
