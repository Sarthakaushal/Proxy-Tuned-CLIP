import pynvml

def initialize_nvml():
    pynvml.nvmlInit()

def shutdown_nvml():
    pynvml.nvmlShutdown()

def get_power_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you have only one GPU
    power_info = pynvml.nvmlDeviceGetPowerUsage(handle)
    return power_info / 1000.0  # Convert to watts
