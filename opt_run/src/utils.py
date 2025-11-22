import sys
import os
import psutil
import torch
import logging

def setup_logging():
    """Configures minimal logging."""
    logging.getLogger("urllib3").setLevel(logging.ERROR)

def inspect_environment():
    """Prints environment statistics."""
    print("--- Inspecting The Environment ---")

    # Check python version
    print("Python Version:")
    print(sys.version)

    # Use os to get CPU cores
    cpu_cores = os.cpu_count()
    print(f"CPU Cores: {cpu_cores}")

    # Check for GPU
    is_gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {is_gpu_available}")

    # Print RAM and storage specs
    # RAM Information
    ram_info = psutil.virtual_memory()
    total_ram_gb = ram_info.total / (1024**3)
    print(f"Total RAM: {total_ram_gb:.2f} GB")

    # Storage Information
    disk_info = psutil.disk_usage('/')
    total_disk_gb = disk_info.total / (1024**3)
    free_disk_gb = disk_info.free / (1024**3)
    print(f"Total Disk Space: {total_disk_gb:.2f} GB")
    print(f"Free Disk Space: {free_disk_gb:.2f} GB")

