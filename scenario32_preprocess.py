import os
import numpy as np
import pandas as pd
from tqdm import tqdm

#路径设置
ROOT = "datasets/scenario33"
CSV_PATH = os.path.join(ROOT, "scenario33_dev.csv")

#读取基站GPS
def load_bs_location():
    gps_path = os.path.join(ROOT, "unit1/GPS_data/gps_location.txt")
    with open(gps_path, "r") as f:
        lines = f.readlines()
    lat = float(lines[0].strip())
    lon = float(lines[1].strip())
    return lat, lon

#经纬度转坐标
def latlon_to_xy(lat, lon, lat_ref, lon_ref):
    R = 6371000  # 地球半径（米）

    dlat = np.radians(lat - lat_ref)
    dlon = np.radians(lon - lon_ref)

    x = R * dlon * np.cos(np.radians(lat_ref))
    y = R * dlat

    return x, y

#预处理函数
def preprocess():

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print("Loading BS location...")
    bs_lat, bs_lon = load_bs_location()

    dx_list = []
    dy_list = []

    print("Processing samples...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):

        # === 1️⃣ 读取 unit2 GPS ===
        unit2_gps_path = os.path.join(ROOT, row["unit2_loc"][2:])

        with open(unit2_gps_path, "r") as f:
            lines = f.readlines()

        ue_lat = float(lines[0].strip())
        ue_lon = float(lines[1].strip())

        # === 2️⃣ 计算相对位移 ===
        dx, dy = latlon_to_xy(ue_lat, ue_lon, bs_lat, bs_lon)

        dx_list.append(dx)
        dy_list.append(dy)

    df["dx"] = dx_list
    df["dy"] = dy_list

    # === 3️⃣ 处理标签 ===
    labels = df["unit1_beam"].values

    df["label"] = df["unit1_beam"] - 1


    # === 4️⃣ 处理图像路径 ===
    df["image_path"] = df["unit1_rgb"].apply(
        lambda x: os.path.join(ROOT, x[2:])
    )

    # === 5️⃣ 标准化 dx, dy ===
    dx_mean = df["dx"].mean()
    dx_std = df["dx"].std()

    dy_mean = df["dy"].mean()
    dy_std = df["dy"].std()

    df["dx"] = (df["dx"] - dx_mean) / dx_std
    df["dy"] = (df["dy"] - dy_mean) / dy_std

    print("dx range:", df["dx"].min(), df["dx"].max())
    print("dy range:", df["dy"].min(), df["dy"].max())

    # === 6️⃣ 保存 ===
    output_path = os.path.join(ROOT, "scenario33_processed.csv")
    df[["image_path", "dx", "dy", "label"]].to_csv(output_path, index=False)

    print("Saved to:", output_path)

#主函数
if __name__ == "__main__":
    preprocess()
