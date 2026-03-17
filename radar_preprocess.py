"""
Radar FFT preprocessing (research-grade)

Pipeline:
ADC → Range Window → Range FFT → Keep positive freq
→ Clutter Removal
→ Doppler Window → Doppler FFT → FFT Shift
→ Angle FFT

Input:
radar ADC (.npy)
shape: (antennas, adc_samples, chirps)

Output:
range_fft
rd_map
ra_map
"""

import argparse
import json
import numpy as np
from pathlib import Path

class RadarPreprocess:
    def __init__(
        self,
        range_fft_size=None,
        doppler_fft_size=None,
        angle_fft_size=None,
    ):
        self.range_fft_size = range_fft_size
        self.doppler_fft_size = doppler_fft_size
        self.angle_fft_size = angle_fft_size

    def load_radar(self, radar_path):
        data = np.load(radar_path)

        if data.ndim != 3:
            raise ValueError("Radar data must be (antenna, adc_samples, chirps)")

        return data

    def apply_range_fft(self, data):
        adc_samples = data.shape[1]
        fft_size = self.range_fft_size or adc_samples

        window = np.hanning(adc_samples)
        window = window[None, :, None]

        data = data * window

        range_fft = np.fft.fft(data, fft_size, axis=1)

        # Keep positive frequencies only.
        range_fft = range_fft[:, : fft_size // 2, :]

        return range_fft

    def remove_clutter(self, data):
        data = data - np.mean(data, axis=2, keepdims=True)
        return data

    def apply_doppler_fft(self, data):
        chirps = data.shape[2]
        fft_size = self.doppler_fft_size or chirps

        window = np.hanning(chirps)
        window = window[None, None, :]

        data = data * window

        doppler_fft = np.fft.fft(data, fft_size, axis=2)
        doppler_fft = np.fft.fftshift(doppler_fft, axes=2)

        return doppler_fft

    def apply_angle_fft(self, data):
        antennas = data.shape[0]
        fft_size = self.angle_fft_size or antennas

        angle_fft = np.fft.fft(data, fft_size, axis=0)
        angle_fft = np.fft.fftshift(angle_fft, axes=0)

        return angle_fft

    def process(self, radar_path):
        data = self.load_radar(radar_path)

        range_fft = self.apply_range_fft(data)
        range_fft = self.remove_clutter(range_fft)

        rd_map = self.apply_doppler_fft(range_fft)
        ra_map = self.apply_angle_fft(range_fft)

        return {
            "range_fft": range_fft,
            "rd_map": rd_map,
            "ra_map": ra_map,
        }


def to_ml_feature(arr, feature_type="log_mag"):
    mag = np.abs(arr).astype(np.float32)

    if feature_type == "mag":
        return mag

    if feature_type == "log_mag":
        return np.log1p(mag)

    raise ValueError(f"Unsupported feature_type: {feature_type}")


def preprocess_directory(
    input_dir,
    output_dir,
    feature_type="log_mag",
    range_fft_size=None,
    doppler_fft_size=None,
    angle_fft_size=None,
    overwrite=False,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {input_path}")

    processor = RadarPreprocess(
        range_fft_size=range_fft_size,
        doppler_fft_size=doppler_fft_size,
        angle_fft_size=angle_fft_size,
    )

    summary = {
        "input_dir": str(input_path.resolve()),
        "output_dir": str(output_path.resolve()),
        "num_files": len(files),
        "feature_type": feature_type,
        "saved_format": "npz",
        "overwrite": overwrite,
    }

    saved_count = 0
    skipped_count = 0

    for idx, radar_file in enumerate(files, start=1):
        out_file = output_path / f"{radar_file.stem}_processed.npz"

        if out_file.exists() and not overwrite:
            skipped_count += 1
            if idx == 1 or idx % 200 == 0 or idx == len(files):
                print(f"[{idx}/{len(files)}] skipped: {out_file.name}")
            continue

        result = processor.process(str(radar_file))

        range_feat = to_ml_feature(result["range_fft"], feature_type)
        rd_feat = to_ml_feature(result["rd_map"], feature_type)
        ra_feat = to_ml_feature(result["ra_map"], feature_type)

        np.savez_compressed(
            out_file,
            range_fft=range_feat,
            rd_map=rd_feat,
            ra_map=ra_feat,
        )

        saved_count += 1
        if idx == 1 or idx % 200 == 0 or idx == len(files):
            print(f"[{idx}/{len(files)}] saved: {out_file.name}")

    summary["saved_files"] = saved_count
    summary["skipped_files"] = skipped_count

    summary_path = output_path / "preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Preprocessing finished.")
    print(f"Output directory: {output_path.resolve()}")
    print(f"Summary file: {summary_path.resolve()}")


def parse_args():
    default_input = "/home/yuhan/beam_prediction_multimodel/datasets/scenario32/unit1/radar_data"
    default_output = "/home/yuhan/beam_prediction_multimodel/datasets/scenario32/unit1/radar_data_processed"

    parser = argparse.ArgumentParser(description="Batch preprocess radar .npy files")

    parser.add_argument("--input-dir", type=str, default=default_input)
    parser.add_argument("--output-dir", type=str, default=default_output)

    parser.add_argument(
        "--feature-type",
        type=str,
        default="log_mag",
        choices=["mag", "log_mag"],
    )

    parser.add_argument("--range-fft-size", type=int, default=None)
    parser.add_argument("--doppler-fft-size", type=int, default=None)
    parser.add_argument("--angle-fft-size", type=int, default=None)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    preprocess_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        feature_type=args.feature_type,
        range_fft_size=args.range_fft_size,
        doppler_fft_size=args.doppler_fft_size,
        angle_fft_size=args.angle_fft_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
