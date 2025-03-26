import requests
import multiprocessing as mp
import os
from utils import log_time, dict_to_pickle, function_logs

file_url = "https://web.ais.dk/aisdata/aisdk-2025-02-14.zip"
file_name = file_url.split("/")[-1]


def download_chunk(url: str, start: int, end: int, file_path: str, index: int) -> None:
    """Download a specific chunk of the file."""
    headers = {"Range": f"bytes={start}-{end}"}

    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()
        with open(f"{file_path}.part{index}", "wb") as file:
            for chunk in response.iter_content(chunk_size=1048576):  # 1 MB chunk size
                file.write(chunk)
    print(f"Chunk {index} completed and saved to {file_path}.part{index}")


def merge_files(file_path: str, num_parts: int, buffer_size: int = 1048576) -> None:
    """Merge all downloaded parts into a single file efficiently."""
    with open(file_path, "wb") as final_file:
        for i in range(num_parts):
            part_path = f"{file_path}.part{i}"
            with open(part_path, "rb") as part_file:
                while True:
                    chunk = part_file.read(buffer_size)
                    if not chunk:
                        break
                    final_file.write(chunk)
            os.remove(part_path)  # Remove part file after merging
    print(f"All parts merged successfully into {file_path}")


@log_time()
def download_file_sequential(
    url: str, filename: str, destination_path: str = "."
) -> None:
    """Download a file in sequential manner from a URL and save it to the local disk."""
    file_path = os.path.join(destination_path, filename)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1048576):
                file.write(chunk)
    print(f"Download completed (Sequential): {file_path}")


@log_time()
def download_file_parallel(
    url: str, filename: str, destination_path: str = ".", num_processes: int = None
) -> None:
    """Download a file in parallel using multiple processes."""

    file_path = os.path.join(destination_path, filename)
    # Make an HTTP HEAD request to fetch file size
    response = requests.head(url)
    file_size = int(response.headers.get("Content-Length", 0))
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Calculate chunk size for each process
    chunk_size = file_size // num_processes

    tasks: list[tuple[str, int, int, str, int]] = []

    # Prepare arguments for each process
    for i in range(num_processes):
        start = i * chunk_size
        end = (i + 1) * chunk_size - 1 if i < num_processes - 1 else file_size - 1
        tasks.append((url, start, end, file_path, i))

    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(download_chunk, tasks)

    # Merge the parts into a single file
    merge_files(file_path, num_processes)
    print(f"Download completed (Parallel) and saved to {file_path}")


if __name__ == "__main__":
    os.makedirs(r".\sequential", exist_ok=True)
    download_file_sequential(file_url, file_name, destination_path=r".\sequential")
    os.makedirs(r".\parallel", exist_ok=True)
    download_file_parallel(
        file_url, file_name, destination_path=r".\parallel", num_processes=8
    )
    os.makedirs(r".\parallel2", exist_ok=True)
    download_file_parallel(
        file_url, file_name, destination_path=r".\parallel2", num_processes=16
    )
    current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    dict_to_pickle(function_logs, rf"logs\{current_file_name}_function_logs.pkl")
