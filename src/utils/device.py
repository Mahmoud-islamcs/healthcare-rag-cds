import logging
import torch

logger = logging.getLogger(__name__)

def get_optimal_device(requested_device: str = "auto") -> str:
    if requested_device in ["cuda", "cpu", "mps"]:
        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return requested_device

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"CUDA GPU detected: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        return "cuda"
    
    logger.info("No CUDA GPU detected. Using CPU.")
    return "cpu"

def get_system_hardware_info() -> dict:
    cuda_avail = torch.cuda.is_available()
    return {
        "cuda_available": cuda_avail,
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "N/A (CPU Mode)",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if cuda_avail else 0.0,
        "torch_version": torch.__version__,
    }
