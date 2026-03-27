import os

POSE_FRAME_SKIP = max(1, int(os.getenv("POSE_FRAME_SKIP", "2")))
SAM2_FRAME_SKIP = max(1, int(os.getenv("SAM2_FRAME_SKIP", "2")))
POSE_MAX_HEIGHT = int(os.getenv("POSE_MAX_HEIGHT", "0"))
POSE_USE_TENSORRT = os.getenv("POSE_USE_TENSORRT", "true").lower() == "true"
POSE_TRT_CACHE_DIR = os.getenv("POSE_TRT_CACHE_DIR", "/app/trt_cache")
POSE_TRT_FP16 = os.getenv("POSE_TRT_FP16", "true").lower() == "true"
