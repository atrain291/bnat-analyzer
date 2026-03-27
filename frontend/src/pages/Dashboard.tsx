import { useState, useEffect, useRef, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, UserPlus, Music, PlayCircle, Clock, CheckCircle2, Loader2, AlertCircle, Trash2 } from "lucide-react";
import { listDancers, createDancer, Dancer } from "../api/dancers";
import { uploadVideo, listPerformances, deletePerformance, PerformanceListItem } from "../api/performances";

const ITEM_TYPES = [
  "Alarippu",
  "Jatiswaram",
  "Shabdam",
  "Varnam",
  "Padam",
  "Tillana",
  "Shlokam",
  "Keerthanam",
  "Thillana",
  "Other",
];

export default function Dashboard() {
  const navigate = useNavigate();
  const [dancers, setDancers] = useState<Dancer[]>([]);
  const [selectedDancer, setSelectedDancer] = useState<number | null>(null);
  const [newDancerName, setNewDancerName] = useState("");
  const [newDancerLevel, setNewDancerLevel] = useState("");
  const [showNewDancer, setShowNewDancer] = useState(false);

  const [itemName, setItemName] = useState("");
  const [itemType, setItemType] = useState("");
  const [talam, setTalam] = useState("");
  const [ragam, setRagam] = useState("");

  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [performances, setPerformances] = useState<PerformanceListItem[]>([]);

  useEffect(() => {
    listDancers().then(setDancers).catch(() => {});
    listPerformances().then(setPerformances).catch(() => {});
  }, []);

  const dancerMap = useMemo(() => new Map(dancers.map((d) => [d.id, d])), [dancers]);

  const handleCreateDancer = async () => {
    if (!newDancerName.trim()) return;
    try {
      const dancer = await createDancer(newDancerName.trim(), newDancerLevel || undefined);
      setDancers((prev) => [...prev, dancer]);
      setSelectedDancer(dancer.id);
      setNewDancerName("");
      setNewDancerLevel("");
      setShowNewDancer(false);
    } catch {
      setError("Failed to create dancer profile");
    }
  };

  const handleUpload = async () => {
    if (!file || !selectedDancer) return;
    setUploading(true);
    setError("");
    setUploadPct(0);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const result = await uploadVideo(
        file,
        selectedDancer,
        itemName || undefined,
        itemType || undefined,
        talam || undefined,
        ragam || undefined,
        setUploadPct,
        controller.signal
      );
      navigate(`/processing/${result.performance_id}`);
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== "CanceledError") {
        setError("Upload failed. Please try again.");
      }
    } finally {
      setUploading(false);
      abortRef.current = null;
    }
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    setUploading(false);
  };

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Delete this performance? This cannot be undone.")) return;
    try {
      await deletePerformance(id);
    } catch {
      // May already be deleted — refresh list either way
    }
    setPerformances((prev) => prev.filter((p) => p.id !== id));
    listPerformances().then(setPerformances).catch(() => {});
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-brand-500">Dashboard</h1>
        <p className="mt-2 text-gray-400">
          Upload a Bharatanatyam practice video for AI-powered form analysis
        </p>
      </div>

      {/* Past Performances */}
      {performances.length > 0 && (
        <section className="rounded-lg bg-gray-800 p-6 space-y-4">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <PlayCircle size={20} /> Past Performances
          </h2>
          <div className="space-y-2">
            {performances.map((p) => {
              const dancerName = dancerMap.get(p.dancer_id)?.name;
              const statusIcon =
                p.status === "complete" ? <CheckCircle2 size={16} className="text-green-400" /> :
                p.status === "failed" ? <AlertCircle size={16} className="text-red-400" /> :
                p.status === "processing" || p.status === "detecting" ? <Loader2 size={16} className="text-brand-400 animate-spin" /> :
                <Clock size={16} className="text-gray-400" />;

              const href =
                p.status === "complete" ? `/review/${p.id}` :
                p.status === "awaiting_selection" ? `/select-dancers/${p.id}` :
                p.status === "failed" ? null :
                `/processing/${p.id}`;

              return (
                <div
                  key={p.id}
                  className={`flex items-center justify-between rounded-lg bg-gray-700/50 px-4 py-3 ${
                    href ? "cursor-pointer hover:bg-gray-700" : ""
                  }`}
                  onClick={() => href && navigate(href)}
                >
                  <div className="flex items-center gap-3">
                    {statusIcon}
                    <div>
                      <div className="text-white font-medium">
                        {p.item_name || p.item_type || "Untitled"}
                        {dancerName && <span className="text-gray-400 font-normal"> — {dancerName}</span>}
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(p.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" })}
                        {p.duration_ms && ` · ${(p.duration_ms / 1000).toFixed(0)}s`}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {p.status === "complete" && p.overall_score !== null && (
                      <div className="text-right">
                        <div className="text-lg font-bold text-brand-400">{p.overall_score}</div>
                        <div className="text-xs text-gray-500">/ 100</div>
                      </div>
                    )}
                    {p.status !== "complete" && p.status !== "failed" && (
                      <span className="text-xs text-gray-400 capitalize">{p.status.replace("_", " ")}</span>
                    )}
                    <button
                      className="p-1.5 rounded text-gray-500 hover:text-red-400 hover:bg-gray-600/50 transition-colors"
                      title="Delete performance"
                      onClick={(e) => handleDelete(p.id, e)}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {error && (
        <div className="rounded-lg bg-red-900/30 border border-red-700 p-4 text-red-300">
          {error}
        </div>
      )}

      {/* Dancer Selection */}
      <section className="rounded-lg bg-gray-800 p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <UserPlus size={20} /> Dancer Profile
        </h2>

        {dancers.length > 0 && (
          <select
            className="w-full rounded bg-gray-700 px-4 py-2 text-white"
            value={selectedDancer ?? ""}
            onChange={(e) => setSelectedDancer(Number(e.target.value) || null)}
          >
            <option value="">Select a dancer...</option>
            {dancers.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name} {d.experience_level ? `(${d.experience_level})` : ""}
              </option>
            ))}
          </select>
        )}

        {!showNewDancer ? (
          <button
            className="text-sm text-brand-400 hover:text-brand-300"
            onClick={() => setShowNewDancer(true)}
          >
            + Create new dancer profile
          </button>
        ) : (
          <div className="flex gap-2 items-end">
            <div className="flex-1">
              <label className="block text-xs text-gray-400 mb-1">Name</label>
              <input
                className="w-full rounded bg-gray-700 px-3 py-2 text-white"
                placeholder="Dancer name"
                value={newDancerName}
                onChange={(e) => setNewDancerName(e.target.value)}
              />
            </div>
            <div className="w-40">
              <label className="block text-xs text-gray-400 mb-1">Level</label>
              <select
                className="w-full rounded bg-gray-700 px-3 py-2 text-white"
                value={newDancerLevel}
                onChange={(e) => setNewDancerLevel(e.target.value)}
              >
                <option value="">Select...</option>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
                <option value="professional">Professional</option>
              </select>
            </div>
            <button
              className="rounded bg-brand-600 px-4 py-2 text-white hover:bg-brand-700"
              onClick={handleCreateDancer}
            >
              Create
            </button>
            <button
              className="rounded bg-gray-600 px-4 py-2 text-white hover:bg-gray-500"
              onClick={() => setShowNewDancer(false)}
            >
              Cancel
            </button>
          </div>
        )}
      </section>

      {/* Dance Metadata */}
      <section className="rounded-lg bg-gray-800 p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Music size={20} /> Performance Details (Optional)
        </h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Item Name</label>
            <input
              className="w-full rounded bg-gray-700 px-3 py-2 text-white"
              placeholder="e.g. Alarippu in Tisra Nadai"
              value={itemName}
              onChange={(e) => setItemName(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Item Type</label>
            <select
              className="w-full rounded bg-gray-700 px-3 py-2 text-white"
              value={itemType}
              onChange={(e) => setItemType(e.target.value)}
            >
              <option value="">Select type...</option>
              {ITEM_TYPES.map((t) => (
                <option key={t} value={t.toLowerCase()}>
                  {t}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Talam</label>
            <input
              className="w-full rounded bg-gray-700 px-3 py-2 text-white"
              placeholder="e.g. Adi, Rupaka, Misra Chapu"
              value={talam}
              onChange={(e) => setTalam(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Ragam</label>
            <input
              className="w-full rounded bg-gray-700 px-3 py-2 text-white"
              placeholder="e.g. Nattai, Kalyani"
              value={ragam}
              onChange={(e) => setRagam(e.target.value)}
            />
          </div>
        </div>
      </section>

      {/* Upload Area */}
      <section className="rounded-lg bg-gray-800 p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Upload size={20} /> Upload Video
        </h2>

        <label
          className="relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-600 p-12 hover:border-brand-500 transition-colors cursor-pointer"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.mov,.avi,.mkv,.webm"
            className="absolute inset-0 overflow-hidden opacity-0 cursor-pointer"
            style={{ width: '100%', height: '100%' }}
            tabIndex={-1}
            onChange={(e) => { setFile(e.target.files?.[0] ?? null); e.target.value = ""; }}
          />
          <Upload size={48} className="text-gray-500 mb-4" />
          {file ? (
            <p className="text-white">
              {file.name}{" "}
              <span className="text-gray-400">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
            </p>
          ) : (
            <>
              <p className="text-gray-400">Drag and drop a video file here, or click to browse</p>
              <p className="text-xs text-gray-500 mt-2">MP4, MOV, AVI, MKV, WebM supported (max 2GB)</p>
            </>
          )}
        </label>

        {uploading && (
          <div className="space-y-2">
            <div className="h-2 rounded-full bg-gray-700 overflow-hidden">
              <div
                className="h-full bg-brand-500 transition-all duration-300"
                style={{ width: `${uploadPct}%` }}
              />
            </div>
            <div className="flex justify-between text-sm text-gray-400">
              <span>Uploading... {uploadPct}%</span>
              <button className="text-red-400 hover:text-red-300" onClick={handleCancel}>
                Cancel
              </button>
            </div>
          </div>
        )}

        <button
          className="w-full rounded-lg bg-brand-600 py-3 text-lg font-semibold text-white hover:bg-brand-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          disabled={!file || !selectedDancer || uploading}
          onClick={handleUpload}
        >
          {uploading ? "Uploading..." : "Analyze Performance"}
        </button>
      </section>
    </div>
  );
}
