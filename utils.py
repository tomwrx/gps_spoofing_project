import pickle
import time
import tracemalloc
import threading as th
import multiprocessing as mp
from functools import wraps


def dict_to_pickle(data: dict, file_name: str) -> None:
    """Save a dictionary to a pickle file."""
    with open(file_name, "wb+") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name: str) -> dict:
    """Load a pickle file and return its content."""
    with open(file_name, "rb") as fh:
        return pickle.load(fh)


# Dictionary to store function performance logs
function_logs: dict[str, any] = {}


def log_time(function_logs: dict = function_logs) -> callable:
    """Decorator to measure execution time, memory usage and log details."""

    def decorator(func: callable) -> callable:
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            execution_type = "sequential"
            if th.current_thread() != th.main_thread():
                execution_type = "multithreading"
            elif mp.current_process().name != "MainProcess":
                execution_type = "multiprocessing"
            elif "num_processes" in kwargs and kwargs["num_processes"] > 1:
                execution_type = "multiprocessing"

            # Start measuring memory usage
            tracemalloc.start()
            start_time = time.perf_counter()

            # Execute the function
            result = func(*args, **kwargs)

            # Stop measuring time and memory
            end_time = time.perf_counter()
            total_time = end_time - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Log details
            function_name = func.__name__
            log_entry = {
                "execution_type": execution_type,
                "execution_time_sec": total_time,
                "current_memory_usage_mb": current / 1024 / 1024,  # MB
                "peak_memory_usage_mb": peak / 1024 / 1024,  # MB
                "args": args,
                "kwargs": kwargs,
            }

            # Initialize log entry list if not already present
            if function_name not in function_logs:
                function_logs[function_name] = []

            # Add log entry
            function_logs[function_name].append(log_entry)

            # Print log info for quick monitoring
            print(f"Function '{function_name}' completed in {total_time:.4f} seconds")
            print(
                f"Current memory usage: {log_entry['current_memory_usage_mb']:.4f} MB; Peak: {log_entry['peak_memory_usage_mb']:.4f} MB"
            )
            return result

        return timeit_wrapper

    return decorator
