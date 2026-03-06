import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, UserPlus, Music } from "lucide-react";
import { listDancers, createDancer, Dancer } from "../api/dancers";
import { uploadVideo } from "../api/performances";

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

        <div
          className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-600 p-12 hover:border-brand-500 transition-colors cursor-pointer"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => document.getElementById("file-input")?.click()}
        >
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
          <input
            id="file-input"
            type="file"
            accept=".mp4,.mov,.avi,.mkv,.webm"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>

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
