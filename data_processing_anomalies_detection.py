import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from functools import partial
from utils import log_time, dict_to_pickle, function_logs


@log_time()
def load_and_preprocess_data_seq() -> pd.DataFrame:
    """Load and preprocess data"""

    irrelevant_statuses = ["Moored", "At anchor", "Not under command"]
    df = pd.read_csv(
        r"data\aisdk-2025-02-14.csv",
        usecols=[
            "# Timestamp",
            "MMSI",
            "Latitude",
            "Longitude",
            "SOG",
            "COG",
            "Navigational status",
        ],
    ).dropna()
    df = df.drop_duplicates()
    df = df[~(df["Navigational status"].isin(irrelevant_statuses))].reset_index(
        drop=True
    )
    df["# Timestamp"] = pd.to_datetime(
        df["# Timestamp"], dayfirst=True, errors="coerce"
    )
    df.rename({"# Timestamp": "Timestamp"}, axis=1, inplace=True)

    return df


def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    irrelevant_statuses = ["Moored", "At anchor", "Not under command"]
    chunk = chunk.dropna().drop_duplicates()
    chunk = chunk[~chunk["Navigational status"].isin(irrelevant_statuses)]
    chunk["Timestamp"] = pd.to_datetime(
        chunk["# Timestamp"], dayfirst=True, errors="coerce"
    )
    chunk.drop("# Timestamp", axis=1, inplace=True)
    return chunk


@log_time()
def load_and_preprocess_data_parallel(
    csv_path: str = r"data\aisdk-2025-02-14.csv", num_processes: int = None
) -> pd.DataFrame:
    if num_processes is None:
        num_processes = mp.cpu_count()

    chunksize = 1000000  # 1 million rows per chunk
    results = []

    with mp.Pool(num_processes) as pool:
        for chunk_result in pool.imap(
            preprocess_chunk,
            pd.read_csv(
                csv_path,
                usecols=[
                    "# Timestamp",
                    "MMSI",
                    "Latitude",
                    "Longitude",
                    "SOG",
                    "COG",
                    "Navigational status",
                ],
                chunksize=chunksize,
            ),
        ):
            results.append(chunk_result)

    df = pd.concat(results, ignore_index=True)
    return df


def calculate_vectorized_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    """
    R = 6371e3  # meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # meters


def angular_difference(angle1: float, angle2: float) -> float:
    """
    Calculate smallest angular difference between two angles (0-360).
    """
    diff = np.abs(angle1 - angle2) % 360
    return np.minimum(diff, 360 - diff)


@log_time()
def flag_anomaly_types_seq(anomalies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags different types of anomalies in the anomalies DataFrame by adding indicator columns.
    """

    anomalies_df["Lat_Long_anomaly"] = 0
    anomalies_df["Location_jump_anomaly"] = 0
    anomalies_df["Reported_speed_anomaly"] = 0
    anomalies_df["Speed_mismatch_anomaly"] = 0
    anomalies_df["Course_anomaly"] = 0

    for index, row in anomalies_df.iterrows():
        if (
            row["Latitude"] < 0
            or row["Longitude"] < 0
            or row["prev_Latitude"] < 0
            or row["prev_Longitude"] < 0
        ):
            anomalies_df.loc[index, "Lat_Long_anomaly"] = 1

        if row["distance_m"] >= 10 * 1852:
            anomalies_df.loc[index, "Location_jump_anomaly"] = 1

        if row["reported_speed_mps"] > 130 * 0.514444:
            anomalies_df.loc[index, "Reported_speed_anomaly"] = 1

        speed_diff_ratio = np.abs(
            row["computed_speed_mps"] - row["reported_speed_mps"]
        ) / (row["reported_speed_mps"] + 1e-6)

        if speed_diff_ratio > 0.7:
            anomalies_df.loc[index, "Speed_mismatch_anomaly"] = 1

        if row["cog_change_deg"] >= 90:
            anomalies_df.loc[index, "Course_anomaly"] = 1

    return anomalies_df


@log_time()
def detect_location_speed_course_anomalies_seq(
    df: pd.DataFrame,
    max_jump_nm: int = 10,
    max_realistic_speed_knots: int = 130,
    speed_tolerance: float = 0.7,
    min_speed_mps: float = 0.514,  # 1 knot
    max_cog_change_deg: float = 90,  # degrees
):
    """
    Detect anomalies based on sudden location jumps, speed inconsistencies,
    and sudden unrealistic changes in course.
    """
    anomalies = []
    threshold_meters = max_jump_nm * 1852
    max_realistic_speed_mps = max_realistic_speed_knots * 0.514444

    for _, group in df.groupby("MMSI"):
        group = group.sort_values("Timestamp").reset_index(drop=True)

        # Previous points
        group["prev_Latitude"] = group["Latitude"].shift(1)
        group["prev_Longitude"] = group["Longitude"].shift(1)
        group["prev_Timestamp"] = group["Timestamp"].shift(1)
        group["prev_SOG"] = group["SOG"].shift(1)
        group["prev_COG"] = group["COG"].shift(1)

        group.dropna(inplace=True)

        # Distance
        group["distance_m"] = calculate_vectorized_distance(
            group["Latitude"].values,
            group["Longitude"].values,
            group["prev_Latitude"].values,
            group["prev_Longitude"].values,
        )

        # Time differences
        group["time_diff_s"] = (
            group["Timestamp"] - group["prev_Timestamp"]
        ).dt.total_seconds()
        group = group[group["time_diff_s"] > 0]

        # Speeds
        group["computed_speed_mps"] = group["distance_m"] / group["time_diff_s"]
        group["reported_speed_mps"] = group["SOG"] * 0.514444

        # Speed anomaly conditions
        condition_distance = group["distance_m"] >= threshold_meters
        condition_unrealistic_speed = (
            group["reported_speed_mps"] > max_realistic_speed_mps
        )

        speed_diff_ratio = np.abs(
            group["computed_speed_mps"] - group["reported_speed_mps"]
        ) / (group["reported_speed_mps"] + 1e-6)
        condition_speed_mismatch = speed_diff_ratio > speed_tolerance
        condition_min_computed_speed = group["computed_speed_mps"] >= min_speed_mps
        condition_negative_lat_lon = (group["Latitude"] < 0) | (group["Longitude"] < 0)

        # Course consistency condition
        group["cog_change_deg"] = angular_difference(group["COG"], group["prev_COG"])
        condition_cog_jump = group["cog_change_deg"] >= max_cog_change_deg

        condition_location_jump = (
            condition_distance
            & condition_min_computed_speed
            & condition_speed_mismatch
            & condition_cog_jump
        )
        combined_conditions = (
            condition_location_jump
            | condition_unrealistic_speed
            | condition_negative_lat_lon
        )
        anomaly_rows = group[combined_conditions]

        if not anomaly_rows.empty:
            anomalies.append(
                anomaly_rows[
                    [
                        "MMSI",
                        "prev_Timestamp",
                        "prev_Latitude",
                        "prev_Longitude",
                        "Timestamp",
                        "Latitude",
                        "Longitude",
                        "distance_m",
                        "computed_speed_mps",
                        "reported_speed_mps",
                        "Navigational status",
                        "COG",
                        "prev_COG",
                        "cog_change_deg",
                    ]
                ]
            )

    anomalies_df = (
        pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame()
    )
    return anomalies_df


def flag_row_anomalies(row: pd.Series) -> pd.Series:
    """
    Flags different types of anomalies in the anomalies DataFrame by adding indicator columns in parallel.
    """
    row["Lat_Long_anomaly"] = int(
        row["Latitude"] < 0
        or row["Longitude"] < 0
        or row["prev_Latitude"] < 0
        or row["prev_Longitude"] < 0
    )
    row["Location_jump_anomaly"] = int(row["distance_m"] >= 10 * 1852)
    row["Reported_speed_anomaly"] = int(row["reported_speed_mps"] > 130 * 0.514444)

    speed_diff_ratio = abs(row["computed_speed_mps"] - row["reported_speed_mps"]) / (
        row["reported_speed_mps"] + 1e-6
    )
    row["Speed_mismatch_anomaly"] = int(speed_diff_ratio > 0.7)
    row["Course_anomaly"] = int(row["cog_change_deg"] >= 90)

    return row


def detect_anomalies_group_parallel(group: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Detect anomalies based on sudden location jumps, speed inconsistencies,
    and sudden unrealistic changes in course in parallel.
    """
    group = group.sort_values("Timestamp").reset_index(drop=True)

    group["prev_Latitude"] = group["Latitude"].shift(1)
    group["prev_Longitude"] = group["Longitude"].shift(1)
    group["prev_Timestamp"] = group["Timestamp"].shift(1)
    group["prev_SOG"] = group["SOG"].shift(1)
    group["prev_COG"] = group["COG"].shift(1)

    group.dropna(inplace=True)

    group["distance_m"] = calculate_vectorized_distance(
        group["Latitude"],
        group["Longitude"],
        group["prev_Latitude"],
        group["prev_Longitude"],
    )

    group["time_diff_s"] = (
        group["Timestamp"] - group["prev_Timestamp"]
    ).dt.total_seconds()
    group = group[group["time_diff_s"] > 0]

    group["computed_speed_mps"] = group["distance_m"] / group["time_diff_s"]
    group["reported_speed_mps"] = group["SOG"] * 0.514444

    speed_diff_ratio = np.abs(
        group["computed_speed_mps"] - group["reported_speed_mps"]
    ) / (group["reported_speed_mps"] + 1e-6)

    condition_distance = group["distance_m"] >= params["threshold_meters"]
    condition_unrealistic_speed = (
        group["reported_speed_mps"] > params["max_realistic_speed_mps"]
    )
    condition_speed_mismatch = speed_diff_ratio > params["speed_tolerance"]
    condition_min_speed = group["computed_speed_mps"] >= params["min_speed_mps"]
    condition_negative_lat_lon = (group["Latitude"] < 0) | (group["Longitude"] < 0)

    group["cog_change_deg"] = angular_difference(group["COG"], group["prev_COG"])
    condition_cog_jump = group["cog_change_deg"] >= params["max_cog_change_deg"]

    condition_location_jump = (
        condition_distance
        & condition_min_speed
        & condition_speed_mismatch
        & condition_cog_jump
    )

    combined_conditions = (
        condition_location_jump
        | condition_unrealistic_speed
        | condition_negative_lat_lon
    )

    return group.loc[
        combined_conditions,
        [
            "MMSI",
            "prev_Timestamp",
            "prev_Latitude",
            "prev_Longitude",
            "Timestamp",
            "Latitude",
            "Longitude",
            "distance_m",
            "computed_speed_mps",
            "reported_speed_mps",
            "Navigational status",
            "COG",
            "prev_COG",
            "cog_change_deg",
        ],
    ]


@log_time()
def detect_anomalies_parallel(
    df: pd.DataFrame, num_processes: int = None
) -> pd.DataFrame:
    if num_processes is None:
        num_processes = mp.cpu_count()

    params = {
        "threshold_meters": 18520,
        "max_realistic_speed_mps": 130 * 0.514444,
        "speed_tolerance": 0.7,
        "min_speed_mps": 0.514,
        "max_cog_change_deg": 90,
    }

    groups = [group for _, group in df.groupby("MMSI")]

    with mp.Pool(processes=num_processes) as pool:
        func = partial(detect_anomalies_group_parallel, params=params)
        results = pool.map(func, groups)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


@log_time()
def flag_anomaly_types_parallel(
    anomalies_df: pd.DataFrame, num_processes: int = None
) -> pd.DataFrame:
    if num_processes is None:
        num_processes = mp.cpu_count()

    with mp.Pool(num_processes) as pool:
        anomalies_df = pd.DataFrame(
            pool.map(flag_row_anomalies, [row for _, row in anomalies_df.iterrows()])
        )

    return anomalies_df


if __name__ == "__main__":
    # Sequential data loading & preprocessing
    anomalies_df_seq = load_and_preprocess_data_seq()
    anomalies_df_seq = detect_location_speed_course_anomalies_seq(anomalies_df_seq)
    anomalies_df_seq = flag_anomaly_types_seq(anomalies_df_seq)
    # Parallel data loading & preprocessing
    for num_processes in [2, 4, 8, 16]:
        anomalies_df_parallel = load_and_preprocess_data_parallel(
            "data/aisdk-2025-02-14.csv", num_processes=num_processes
        )
        anomalies_df_parallel = detect_anomalies_parallel(
            anomalies_df_parallel, num_processes=num_processes
        )
        anomalies_df_parallel = flag_anomaly_types_parallel(
            anomalies_df_parallel, num_processes=num_processes
        )
    print("Anomalies detected and logged successfully.")
    current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    dict_to_pickle(function_logs, rf"logs\{current_file_name}_function_logs.pkl")
