import os
import torch
import torch.distributed as dist
from training.utils.common import print0
from training.utils.logging import  setup_default_logging
import logging

setup_default_logging()
logger = logging.getLogger(__name__)



def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_initialized() -> bool:
    """Returns True if the distributed environment is initialized."""
    return dist.is_available() and dist.is_initialized()


def cleanup():
    """Destroys the process group."""
    if is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Returns True if the current process is rank 0 or if DDP is off."""
    if not is_initialized():
        return True
    return dist.get_rank() == 0


def reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    """Safely synchronizes a tensor across all GPUs."""
    if not is_initialized():
        return tensor

    reduced = tensor.clone()
    dist.all_reduce(reduced, op=op)
    # Important: Do not divide by world_size here if you want SUM reduction as default.
    return reduced


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def get_rank():
    """Returns current process rank. Defaults to 0 if DDP is off."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_dist_info():
    if is_ddp_requested():
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def autodetect_device_type():
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type=None):
    if not device_type:
        device_type = autodetect_device_type()

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"

    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    if device_type == "cuda":
        torch.set_float32_matmul_precision(
            "high")  # uses tf32 instead of fp32 for matmuls, see https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html

    _is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    if _is_ddp_requested and device_type == "cuda":
        print("settingup cuda")
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return _is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device