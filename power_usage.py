import subprocess
import time
from config import TrainingCFG
# Function to get GPU power usage
def get_gpu_power(exit_flag, buffer):
    while not exit_flag.is_set():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
            power_usage = result.stdout.decode('utf-8').strip()
            buffer.append(power_usage.split('\n')[TrainingCFG.device_num])
            time.sleep(.000011)
        except Exception as e:
            print("Error getting GPU power usage:", e)
        
