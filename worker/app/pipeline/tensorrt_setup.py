import logging
import os

logger = logging.getLogger(__name__)

_patched = False


def enable_tensorrt():
    """Monkey-patch rtmlib to use TensorRT Execution Provider via ONNX Runtime.

    ORT's TensorRT EP compiles ONNX subgraphs into TensorRT engines on first run
    (cached to disk), then uses them for subsequent inference. Falls back to
    CUDAExecutionProvider for unsupported ops automatically.
    """
    global _patched
    if _patched:
        return

    from app.pipeline.pose_config import POSE_TRT_CACHE_DIR, POSE_TRT_FP16

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" not in available:
            logger.warning("TensorrtExecutionProvider not available in onnxruntime, "
                           f"available providers: {available}")
            _patched = True
            return
    except Exception as e:
        logger.warning(f"Could not check onnxruntime providers: {e}")
        _patched = True
        return

    os.makedirs(POSE_TRT_CACHE_DIR, exist_ok=True)

    try:
        import onnxruntime as ort
        from rtmlib.tools.base import BaseTool
        _original_init = BaseTool.__init__

        trt_options = {
            "trt_engine_cache_enable": "1",
            "trt_engine_cache_path": POSE_TRT_CACHE_DIR,
            "trt_fp16_enable": "1" if POSE_TRT_FP16 else "0",
        }
        trt_providers = [
            ("TensorrtExecutionProvider", trt_options),
            ("CUDAExecutionProvider", {}),
        ]

        def _patched_init(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)

            if not hasattr(self, "session") or self.session is None:
                return
            if not hasattr(self, "backend") or self.backend != "onnxruntime":
                return
            if not hasattr(self, "onnx_model"):
                return

            try:
                new_session = ort.InferenceSession(self.onnx_model, providers=trt_providers)
                if new_session.get_inputs() is not None:
                    self.session = new_session
                    logger.info(f"TensorRT EP enabled (fp16={POSE_TRT_FP16}, cache={POSE_TRT_CACHE_DIR})")
                else:
                    logger.warning("TensorRT session created but has no inputs, keeping CUDA EP")
            except Exception as e:
                logger.warning(f"Failed to create TensorRT session, keeping CUDA EP: {e}")

        BaseTool.__init__ = _patched_init
        _patched = True
        logger.info("rtmlib patched for TensorRT Execution Provider")

    except Exception as e:
        logger.warning(f"Failed to patch rtmlib for TensorRT: {e}")
        _patched = True
