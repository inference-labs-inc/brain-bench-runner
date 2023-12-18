import platform
import psutil
import logging
import subprocess

# Setting up logging
logging.basicConfig(level=logging.INFO)

def get_sysctl_value(key):
    try:
        return subprocess.check_output(["sysctl", key], text=True).split(": ")[1].strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting sysctl value for {key}: {e}")
        return None



def get_cpu_frequency():
    tbfrequency = get_sysctl_value("hw.tbfrequency")
    clockrate_hz = get_sysctl_value("kern.clockrate")
    if tbfrequency and clockrate_hz:
        try:
            # Extract the hz value from clockrate
            hz_value = int(clockrate_hz.split(",")[0].split("hz =")[1])
            return (int(tbfrequency) * hz_value) / 10**9
        except IndexError as e:
            logging.error(f"Error processing clockrate value: {e}")
            return None

def get_machine_name():
    machine_info = get_system_info()
    machine_name = machine_info["system"] + '-' + machine_info["processor"] + '-' + str(machine_info["cpu_freq"]) + 'Ghz-' + str(int(machine_info["memory_total"] / (1024 ** 3))) + 'GB'
    return machine_name.replace('.', '')

def get_system_info():
    system_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "cpu_freq": get_cpu_frequency(),
        "memory_total": psutil.virtual_memory().total
    }
    return system_info

def log_system_info():
    # Log basic system information
    logging.info(f"System: {platform.system()}")
    logging.info(f"Release: {platform.release()}")
    logging.info(f"Version: {platform.version()}")
    logging.info(f"Machine: {platform.machine()}")
    logging.info(f"Processor: {platform.processor()}")

    # CPU details
    try:
        logging.info(f"Physical cores: {psutil.cpu_count(logical=False)}")
        logging.info(f"Total cores: {psutil.cpu_count(logical=True)}")
        try:
            cpu_freq = psutil.cpu_freq()
        except:
            cpu_freq = get_cpu_frequency()
        logging.info(f"CPU Frequency: {cpu_freq:.2f} Mhz")
        logging.info(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    except Exception as e:
        logging.error(f"Error getting CPU details: {e}")

    # Memory Information
    svmem = psutil.virtual_memory()
    logging.info(f"Total Memory: {svmem.total / (1024 ** 3):.2f} GB")
    logging.info(f"Available Memory: {svmem.available / (1024 ** 3):.2f} GB")
    logging.info(f"Used Memory: {svmem.used / (1024 ** 3):.2f} GB")
    logging.info(f"Memory Usage: {svmem.percent}%")

    # Disk Information
    partitions = psutil.disk_partitions()
    for partition in partitions:
        logging.info(f"Device: {partition.device}")
        logging.info(f"\tMountpoint: {partition.mountpoint}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            continue
        logging.info(f"\tTotal Size: {partition_usage.total / (1024 ** 3):.2f} GB")
        logging.info(f"\tUsed: {partition_usage.used / (1024 ** 3):.2f} GB")
        logging.info(f"\tFree: {partition_usage.free / (1024 ** 3):.2f} GB")
        logging.info(f"\tPercentage: {partition_usage.percent}%")
