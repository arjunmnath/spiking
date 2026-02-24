import os
import torch
import torch.distributed as dist

def is_initialized():
    """Returns True if the distributed environment is initialized."""
    return dist.is_available() and dist.is_initialized()

def init_distributed():
    """Initializes the distributed environment via torchrun."""
    if "WORLD_SIZE" not in os.environ:
        return False
    
    # Check for correct backend
    if torch.cuda.is_available():
        backend = "nccl"
    elif torch.backends.mps.is_available():
        backend = "gloo" # NCCL not on MPS
    else:
        backend = "gloo"
        
    dist.init_process_group(backend=backend)
    
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    
    return True

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