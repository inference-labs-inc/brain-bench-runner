#!/usr/bin/env python3

import threading
import time
import psutil
import tracemalloc
import cProfile
from util.system_stats import get_system_info


def monitor_resources(interval=1, stop_event=threading.Event()):
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().used
        print(f"Resource Usage - CPU: {cpu_usage}%, Memory: {memory_usage / (1024 ** 2):.2f} MB")
        time.sleep(interval)

def sample_function(execution_time=10):
    # Replace this with the actual work of your function
    time.sleep(execution_time)

import json
import datetime

machine_info = get_system_info()
machine_name = machine_info["system"] + '-' + machine_info["processor"] + '-' + str(machine_info["cpu_freq"]) + 'Ghz-' + str(machine_info["memory_total"] / (1024 ** 3)) + 'GB'

def run_with_monitoring():
    def decorator(func):
        def wrapper(*args, **kwargs):
            wrapper.__name__ = "run_with_monitoring(" + str(func.__name__) + ")"
            print(f"Running function {func.__name__} with resource monitoring...")
            tracemalloc.start()
            profiler = cProfile.Profile()
            stop_event = threading.Event()
            monitor_thread = threading.Thread(target=monitor_resources, args=(1, stop_event))

            start_time = time.time()
            try:
                profiler.enable()
                monitor_thread.start()
                result = func(*args, **kwargs)
                profiler.disable()
            finally:
                stop_event.set()
                monitor_thread.join()
                print(f"Function {func.__name__} has completed execution.")
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
                profiler.print_stats()

                # Output JSON
                execution_time = time.time() - start_time
                output = {
                    "meta": {
                        "lastUpdated": datetime.datetime.now().isoformat()
                    },
                    "frameworks": {
                        "ezkl": {
                            machine_name: {
                                func.__name__: {
                                    "name": func.__name__,
                                    "results": [
                                        {
                                            "metrics": {
                                                "memory_usage_bytes": current,
                                                "peak_memory_usage_bytes": peak
                                            },
                                            "name": func.__name__,
                                            "time": {
                                                "secs": execution_time
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
                print(json.dumps(output, indent=4))
            return result
        return wrapper
    return decorator
