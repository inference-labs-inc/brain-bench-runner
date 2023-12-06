#!/usr/bin/env python3

import threading
import time
import psutil

def monitor_resources(interval=1, stop_event=threading.Event()):
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().used
        print(f"Resource Usage - CPU: {cpu_usage}%, Memory: {memory_usage / (1024 ** 2):.2f} MB")
        time.sleep(interval)

def sample_function(execution_time=10):
    # Replace this with the actual work of your function
    time.sleep(execution_time)

def run_with_monitoring(func, *args, interval=1):
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(interval, stop_event))

    try:
        monitor_thread.start()
        func(*args)
    finally:
        stop_event.set()
        monitor_thread.join()
