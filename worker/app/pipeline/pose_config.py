import os

POSE_FRAME_SKIP = max(1, int(os.getenv("POSE_FRAME_SKIP", "2")))
SAM2_FRAME_SKIP = max(1, int(os.getenv("SAM2_FRAME_SKIP", "2")))
POSE_MAX_HEIGHT = int(os.getenv("POSE_MAX_HEIGHT", "0"))
POSE_USE_TENSORRT = os.getenv("POSE_USE_TENSORRT", "true").lower() == "true"
POSE_TRT_CACHE_DIR = os.getenv("POSE_TRT_CACHE_DIR", "/app/trt_cache")
POSE_TRT_FP16 = os.getenv("POSE_TRT_FP16", "true").lower() == "true"
REID_ENABLED = os.getenv("REID_ENABLED", "true").lower() == "true"
REID_MODEL_PATH = os.getenv("REID_MODEL_PATH", "/app/trt_cache/osnet_x025.onnx")

# Tracker thresholds (tunable without code changes)
TRACKER_RESEED_IDENTITY_THRESHOLD = float(os.getenv("TRACKER_RESEED_IDENTITY_THRESHOLD", "0.4"))
TRACKER_COHERENCE_THRESHOLD = float(os.getenv("TRACKER_COHERENCE_THRESHOLD", "0.4"))
TRACKER_FORMATION_RATIO_LIMIT = float(os.getenv("TRACKER_FORMATION_RATIO_LIMIT", "3.0"))
TRACKER_REID_BASE_THRESHOLD = float(os.getenv("TRACKER_REID_BASE_THRESHOLD", "0.5"))
