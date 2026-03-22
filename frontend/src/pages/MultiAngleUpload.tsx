import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { UserPlus, Music, Camera, X, Plus } from "lucide-react";
import { listDancers, createDancer, Dancer } from "../api/dancers";
import { uploadMultiAngle } from "../api/multiAngle";

const ITEM_TYPES = [
  "Alarippu", "Jatiswaram", "Shabdam", "Varnam", "Padam",
  "Tillana", "Shlokam", "Keerthanam", "Thillana", "Other",
];

const DEFAULT_LABELS = ["Front", "Side"];

export default function MultiAngleUpload() {
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

  const [files, setFiles] = useState<{ file: File; label: string }[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadPct, setUploadPct] = useState(0);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    listDancers().then(setDancers).catch(() => {});
  }, []);

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

  const addFile = (file: File) => {
    const defaultLabel = DEFAULT_LABELS[files.length] || `Camera ${files.length + 1}`;
    setFiles((prev) => [...prev, { file, label: defaultLabel }]);
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const updateLabel = (idx: number, label: string) => {
    setFiles((prev) => prev.map((f, i) => (i === idx ? { ...f, label } : f)));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files);
    dropped.forEach(addFile);
  };

  const handleUpload = async () => {
    if (files.length < 2 || !selectedDancer) return;
    setUploading(true);
    setError("");
    setUploadPct(0);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const result = await uploadMultiAngle(
        files.map((f) => f.file),
        selectedDancer,
        files.map((f) => f.label),
        itemName || undefined,
        itemType || undefined,
        talam || undefined,
        ragam || undefined,
        setUploadPct,
        controller.signal,
      );
      navigate(`/multi-angle/processing/${result.group_id}`);
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

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-brand-500">Multi-Angle Upload</h1>
        <p className="mt-2 text-gray-400">
          Upload videos of the same performance from different camera angles for combined analysis
        </p>
      </div>

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
            <button className="rounded bg-brand-600 px-4 py-2 text-white hover:bg-brand-700" onClick={handleCreateDancer}>Create</button>
            <button className="rounded bg-gray-600 px-4 py-2 text-white hover:bg-gray-500" onClick={() => setShowNewDancer(false)}>Cancel</button>
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
            <input className="w-full rounded bg-gray-700 px-3 py-2 text-white" placeholder="e.g. Alarippu in Tisra Nadai" value={itemName} onChange={(e) => setItemName(e.target.value)} />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Item Type</label>
            <select className="w-full rounded bg-gray-700 px-3 py-2 text-white" value={itemType} onChange={(e) => setItemType(e.target.value)}>
              <option value="">Select type...</option>
              {ITEM_TYPES.map((t) => (<option key={t} value={t.toLowerCase()}>{t}</option>))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Talam</label>
            <input className="w-full rounded bg-gray-700 px-3 py-2 text-white" placeholder="e.g. Adi, Rupaka" value={talam} onChange={(e) => setTalam(e.target.value)} />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Ragam</label>
            <input className="w-full rounded bg-gray-700 px-3 py-2 text-white" placeholder="e.g. Nattai, Kalyani" value={ragam} onChange={(e) => setRagam(e.target.value)} />
          </div>
        </div>
      </section>

      {/* Multi-file Upload Area */}
      <section className="rounded-lg bg-gray-800 p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Camera size={20} /> Upload Videos (2+ angles)
        </h2>

        {/* Added files */}
        {files.map((f, idx) => (
          <div key={idx} className="flex items-center gap-3 rounded-lg bg-gray-700/50 px-4 py-3">
            <Camera size={18} className="text-brand-400 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-white text-sm truncate">{f.file.name} <span className="text-gray-400">({(f.file.size / 1024 / 1024).toFixed(1)} MB)</span></p>
            </div>
            <input
              type="text"
              value={f.label}
              onChange={(e) => updateLabel(idx, e.target.value)}
              className="w-32 rounded bg-gray-600 px-2 py-1 text-sm text-white border border-gray-500 focus:border-brand-500 focus:outline-none"
              placeholder="Camera label"
            />
            <button onClick={() => removeFile(idx)} className="text-gray-400 hover:text-red-400">
              <X size={16} />
            </button>
          </div>
        ))}

        {/* Drop zone for adding more */}
        <div
          className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-600 p-8 hover:border-brand-500 transition-colors cursor-pointer"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => document.getElementById("ma-file-input")?.click()}
        >
          <Plus size={32} className="text-gray-500 mb-2" />
          {files.length === 0 ? (
            <>
              <p className="text-gray-400">Drop video files here, or click to browse</p>
              <p className="text-xs text-gray-500 mt-1">Add at least 2 videos from different camera angles</p>
            </>
          ) : (
            <p className="text-gray-400 text-sm">Add another camera angle...</p>
          )}
          <input
            id="ma-file-input"
            type="file"
            accept=".mp4,.mov,.avi,.mkv,.webm"
            className="hidden"
            multiple
            onChange={(e) => {
              if (e.target.files) Array.from(e.target.files).forEach(addFile);
              e.target.value = "";
            }}
          />
        </div>

        {files.length > 0 && files.length < 2 && (
          <p className="text-amber-400 text-sm">Add at least one more video from a different angle</p>
        )}

        {uploading && (
          <div className="space-y-2">
            <div className="h-2 rounded-full bg-gray-700 overflow-hidden">
              <div className="h-full bg-brand-500 transition-all duration-300" style={{ width: `${uploadPct}%` }} />
            </div>
            <div className="flex justify-between text-sm text-gray-400">
              <span>Uploading {files.length} videos... {uploadPct}%</span>
              <button className="text-red-400 hover:text-red-300" onClick={handleCancel}>Cancel</button>
            </div>
          </div>
        )}

        <button
          className="w-full rounded-lg bg-brand-600 py-3 text-lg font-semibold text-white hover:bg-brand-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          disabled={files.length < 2 || !selectedDancer || uploading}
          onClick={handleUpload}
        >
          {uploading ? "Uploading..." : `Analyze ${files.length} Angles`}
        </button>
      </section>
    </div>
  );
}
