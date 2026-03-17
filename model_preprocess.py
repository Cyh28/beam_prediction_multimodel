import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from radar_preprocess import RadarPreprocess, to_ml_feature

# ==============================
# 路径配置
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, "datasets", "scenario33")

CSV_PATH = os.path.join(ROOT, "scenario33_dev.csv")

RADAR_INPUT_DIR = os.path.join(ROOT, "unit1/radar_data")
RADAR_OUTPUT_DIR = os.path.join(ROOT, "unit1/radar_data_processed")

OUTPUT_CSV = os.path.join(ROOT, "scenario33_processed.csv")


# ==============================
# 读取基站 GPS
# ==============================

def load_bs_location():

    gps_path = os.path.join(ROOT, "unit1/GPS_data/gps_location.txt")

    with open(gps_path, "r") as f:
        lines = f.readlines()

    lat = float(lines[0].strip())
    lon = float(lines[1].strip())

    return lat, lon


# ==============================
# 经纬度转平面坐标
# ==============================

def latlon_to_xy(lat, lon, lat_ref, lon_ref):

    R = 6371000

    dlat = np.radians(lat - lat_ref)
    dlon = np.radians(lon - lon_ref)

    x = R * dlon * np.cos(np.radians(lat_ref))
    y = R * dlat

    return x, y


# ==============================
# Radar 处理
# ==============================

def preprocess_radar():

    processor = RadarPreprocess()

    input_path = Path(RADAR_INPUT_DIR)
    output_path = Path(RADAR_OUTPUT_DIR)

    output_path.mkdir(parents=True, exist_ok=True)

    radar_files = sorted(input_path.glob("*.npy"))

    radar_map = {}

    print("Processing radar files...")

    for radar_file in tqdm(radar_files):

        result = processor.process(str(radar_file))

        range_feat = to_ml_feature(result["range_fft"])
        rd_feat = to_ml_feature(result["rd_map"])
        ra_feat = to_ml_feature(result["ra_map"])

        out_file = output_path / f"{radar_file.stem}_processed.npz"

        np.savez_compressed(
            out_file,
            range_fft=range_feat,
            rd_map=rd_feat,
            ra_map=ra_feat
        )

        radar_map[radar_file.stem] = str(out_file)

    return radar_map


# ==============================
# 主 preprocessing
# ==============================

def preprocess():

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print("Loading BS location...")
    bs_lat, bs_lon = load_bs_location()

    # ========= Radar preprocessing =========
    radar_map = preprocess_radar()

    dx_list = []
    dy_list = []
    radar_paths = []

    print("Processing samples...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        # ======================
        # 1️⃣ 读取 UE GPS
        # ======================

        unit2_gps_path = os.path.join(ROOT, row["unit2_loc"][2:])

        with open(unit2_gps_path, "r") as f:
            lines = f.readlines()

        ue_lat = float(lines[0].strip())
        ue_lon = float(lines[1].strip())

        # ======================
        # 2️⃣ 计算 dx dy
        # ======================

        dx, dy = latlon_to_xy(ue_lat, ue_lon, bs_lat, bs_lon)

        dx_list.append(dx)
        dy_list.append(dy)

        # ======================
        # 3️⃣ radar path
        # ======================

        radar_name = Path(row["unit1_radar"][2:]).stem

        if radar_name not in radar_map:
            raise RuntimeError(f"Radar file missing: {radar_name}")

        radar_paths.append(radar_map[radar_name])

    df["dx"] = dx_list
    df["dy"] = dy_list

    # ======================
    # 标签
    # ======================

    df["label"] = df["unit1_beam"] - 1

    # ======================
    # image path
    # ======================

    df["image_path"] = df["unit1_rgb"].apply(
        lambda x: os.path.join(ROOT, x[2:])
    )

    # ======================
    # radar path
    # ======================

    df["radar_path"] = radar_paths

    # ======================
    # 标准化 GPS
    # ======================

    dx_mean = df["dx"].mean()
    dx_std = df["dx"].std()

    dy_mean = df["dy"].mean()
    dy_std = df["dy"].std()

    df["dx"] = (df["dx"] - dx_mean) / dx_std
    df["dy"] = (df["dy"] - dy_mean) / dy_std

    print("dx range:", df["dx"].min(), df["dx"].max())
    print("dy range:", df["dy"].min(), df["dy"].max())

    # ======================
    # 保存 CSV
    # ======================

    df[
        [
            "image_path",
            "radar_path",
            "dx",
            "dy",
            "label"
        ]
    ].to_csv(OUTPUT_CSV, index=False)

    print("Saved CSV to:", OUTPUT_CSV)


# ==============================

if __name__ == "__main__":
    preprocess()

