import os
import platform
import socket

import psutil
import torch


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {
        "available": True,
        "count": num_devices,
        "names": [],
        "memory_gb": []
    }

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / (1024**3))

    info["cuda_version"] = torch.version.cuda or "unknown"
    return info

def get_system_info():
    """Get system information."""
    info = {}

    # Basic system info
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__

    # CPU and memory
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)

    # User and environment
    info['user'] = os.environ.get('USER', 'unknown')
    info['nanochat_base_dir'] = os.environ.get('NANOCHAT_BASE_DIR', 'out')
    info['working_dir'] = os.getcwd()

    return info
