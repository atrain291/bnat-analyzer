"""Lightweight person Re-ID using OSNet-x0.25 ONNX model.

Produces 512-dim L2-normalized embeddings per person crop for identity matching.
Uses onnxruntime-gpu (already present in worker container).
"""
import logging
import os

import cv2
import numpy as np
import onnxruntime as ort

from app.pipeline.pose_config import REID_ENABLED, REID_MODEL_PATH

logger = logging.getLogger(__name__)

INPUT_HEIGHT = 256
INPUT_WIDTH = 128
EMBED_DIM = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ReIDExtractor:
    _instance = None

    def __new__(cls, model_path: str | None = None):
        if cls._instance is not None:
            return cls._instance
        inst = super().__new__(cls)
        cls._instance = inst
        return inst

    def __init__(self, model_path: str | None = None):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.session = None

        if not REID_ENABLED:
            logger.info("Re-ID disabled via REID_ENABLED=false")
            return

        path = model_path or REID_MODEL_PATH
        if not os.path.exists(path):
            logger.warning(f"Re-ID model not found at {path}. Re-ID will be unavailable. "
                           f"Model should be baked in at build time via export_reid_model.py.")
            return

        providers = []
        if ort.get_available_providers():
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

        try:
            self.session = ort.InferenceSession(path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Re-ID model loaded: OSNet-x0.25 ({path}), providers={self.session.get_providers()}")
        except Exception as e:
            logger.warning(f"Failed to load Re-ID model: {e}. Re-ID will be unavailable.")
            self.session = None

    def extract(self, frame_bgr: np.ndarray, bbox: tuple, normalized: bool = True) -> np.ndarray | None:
        """Extract 512-dim Re-ID embedding from a person crop.

        Args:
            frame_bgr: Full frame in BGR format (HxWx3).
            bbox: (x_min, y_min, x_max, y_max) in normalized [0,1] coords if normalized=True,
                  or pixel coords if False.

        Returns 512-dim L2-normalized float32 vector, or None on failure.
        """
        if self.session is None:
            return None

        h, w = frame_bgr.shape[:2]

        if normalized:
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
        else:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        crop = frame_bgr[y1:y2, x1:x2]

        try:
            # Resize to model input
            crop_resized = cv2.resize(crop, (INPUT_WIDTH, INPUT_HEIGHT))
            # BGR -> RGB, normalize
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop_rgb = (crop_rgb - IMAGENET_MEAN) / IMAGENET_STD
            # HWC -> CHW, add batch dim
            blob = np.transpose(crop_rgb, (2, 0, 1))[np.newaxis].astype(np.float32)

            outputs = self.session.run(None, {self.input_name: blob})
            embedding = outputs[0].flatten().astype(np.float32)

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            logger.debug(f"Re-ID extraction failed: {e}")
            return None


def cosine_similarity(emb_a: np.ndarray | None, emb_b: np.ndarray | None) -> float:
    """Cosine similarity between two L2-normalized embeddings. Returns 0.5 if either is None."""
    if emb_a is None or emb_b is None:
        return 0.5
    return float(np.dot(emb_a, emb_b))


def merge_embeddings(existing: np.ndarray, new: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """EMA update of Re-ID embedding, then re-normalize."""
    merged = existing * (1 - alpha) + new * alpha
    norm = np.linalg.norm(merged)
    if norm > 1e-6:
        merged = merged / norm
    return merged
