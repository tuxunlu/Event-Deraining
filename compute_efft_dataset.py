#!/usr/bin/env python3
"""Compute EFFT for EventRain KITTI Dataset"""

import os
import numpy as np
from efft import Stimulus, Stimuli, eFFT
from pathlib import Path
import json
from tqdm import tqdm
import time


class EventFFTProcessor:

    def __init__(self,
                 data_root="data/eventrain_KITTI/synthetic/synthetic_KITTI/synthetic",
                 output_root="efft_results",
                 frame_size=256):
        self.data_root = data_root
        self.output_root = output_root
        self.frame_size = frame_size
        self.orig_width = 460
        self.orig_height = 352

        Path(output_root).mkdir(parents=True, exist_ok=True)

        print(f"Initialized EventFFTProcessor:")
        print(f"  Data root: {data_root}")
        print(f"  Output root: {output_root}")
        print(f"  Frame size: {frame_size}x{frame_size}")
        print(f"  Original resolution: {self.orig_width}x{self.orig_height}")

    def downsample_coordinates(self, x, y):
        x_scaled = (x * self.frame_size / self.orig_width).astype(np.int32)
        y_scaled = (y * self.frame_size / self.orig_height).astype(np.int32)
        x_scaled = np.clip(x_scaled, 0, self.frame_size - 1)
        y_scaled = np.clip(y_scaled, 0, self.frame_size - 1)
        return x_scaled, y_scaled

    def process_time_window(self, npz_file):
        data = np.load(npz_file)
        x = data['x']
        y = data['y']
        t = data['t']
        p = data['p']
        data.close()

        n_events = len(x)
        x_scaled, y_scaled = self.downsample_coordinates(x, y)
        states = (p == 1)

        efft_instance = eFFT(self.frame_size)
        efft_instance.initialize()

        events = Stimuli()
        for i in range(n_events):
            stimulus = Stimulus(int(y_scaled[i]), int(x_scaled[i]), bool(states[i]))
            events.append(stimulus)

        start_time = time.time()
        efft_instance.update(events)
        processing_time = time.time() - start_time

        fft_result = efft_instance.get_fft()

        stats = {
            'n_events': int(n_events),
            'n_on_events': int(np.sum(p == 1)),
            'n_off_events': int(np.sum(p == -1)),
            'processing_time_ms': processing_time * 1000,
            'time_range_us': [int(t.min()), int(t.max())],
            'duration_ms': float((t.max() - t.min()) / 1000),
        }

        return fft_result, stats

    def save_fft_result(self, fft_result, output_path):
        np.savez_compressed(
            output_path,
            fft_complex=fft_result,
            fft_magnitude=np.abs(fft_result),
            fft_phase=np.angle(fft_result),
            fft_real=np.real(fft_result),
            fft_imag=np.imag(fft_result)
        )

    def process_directory(self, input_dir, output_dir, desc="Processing"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        npz_files = sorted(Path(input_dir).glob("*.npz"))

        if len(npz_files) == 0:
            print(f"Warning: No .npz files found in {input_dir}")
            return []

        all_stats = []

        for npz_file in tqdm(npz_files, desc=desc):
            try:
                fft_result, stats = self.process_time_window(str(npz_file))
                output_filename = npz_file.stem
                output_path = os.path.join(output_dir, output_filename)
                self.save_fft_result(fft_result, output_path)
                stats['filename'] = npz_file.name
                all_stats.append(stats)
            except Exception as e:
                print(f"\nError processing {npz_file}: {e}")
                continue

        return all_stats

    def process_rainfall_intensity(self, rainfall_mm):
        rainfall_dir = f"{rainfall_mm}mm"
        input_dir = os.path.join(self.data_root, "merge_data", rainfall_dir)
        output_dir = os.path.join(self.output_root, "merge_data", rainfall_dir)

        if not os.path.exists(input_dir):
            print(f"Warning: Directory not found: {input_dir}")
            return None

        print(f"\nProcessing rainfall intensity: {rainfall_mm}mm")
        stats = self.process_directory(input_dir, output_dir, desc=f"  {rainfall_mm}mm")
        return stats

    def process_raw_data(self):
        input_dir = os.path.join(self.data_root, "raw_data")
        output_dir = os.path.join(self.output_root, "raw_data")

        if not os.path.exists(input_dir):
            print(f"Warning: Directory not found: {input_dir}")
            return None

        print(f"\nProcessing raw_data (clean events)")
        stats = self.process_directory(input_dir, output_dir, desc="  raw_data")
        return stats

    def process_all(self):
        print("\nComputing EFFT for EventRain KITTI Dataset")
        start_time = time.time()

        merge_data_dir = os.path.join(self.data_root, "merge_data")
        rainfall_dirs = sorted([d.name for d in Path(merge_data_dir).iterdir() if d.is_dir()])
        rainfall_intensities = [int(d.replace('mm', '')) for d in rainfall_dirs]

        print(f"\nFound {len(rainfall_intensities)} rainfall intensities:")
        print(f"  {rainfall_intensities}")

        metadata = {
            'frame_size': self.frame_size,
            'original_resolution': [self.orig_width, self.orig_height],
            'rainfall_intensities': rainfall_intensities,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'merge_data': {},
            'raw_data': None
        }

        for rainfall in rainfall_intensities:
            stats = self.process_rainfall_intensity(rainfall)
            if stats:
                metadata['merge_data'][f'{rainfall}mm'] = {
                    'n_files': len(stats),
                    'total_events': sum(s['n_events'] for s in stats),
                    'total_processing_time_s': sum(s['processing_time_ms'] for s in stats) / 1000,
                    'file_stats': stats
                }

        raw_stats = self.process_raw_data()
        if raw_stats:
            metadata['raw_data'] = {
                'n_files': len(raw_stats),
                'total_events': sum(s['n_events'] for s in raw_stats),
                'total_processing_time_s': sum(s['processing_time_ms'] for s in raw_stats) / 1000,
                'file_stats': raw_stats
            }

        total_time = time.time() - start_time
        metadata['total_processing_time_s'] = total_time

        metadata_path = os.path.join(self.output_root, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nProcessing complete")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Results saved to: {self.output_root}")
        print(f"Metadata saved to: {metadata_path}")

        return metadata


def main():
    processor = EventFFTProcessor(
        data_root="data/eventrain_KITTI/synthetic/synthetic_KITTI/synthetic",
        output_root="efft_results",
        frame_size=256
    )

    metadata = processor.process_all()

    print("\nSummary:")
    if metadata['raw_data']:
        print(f"  Raw data: {metadata['raw_data']['n_files']} files, "
              f"{metadata['raw_data']['total_events']:,} events")

    print(f"  Merge data:")
    for rainfall, data in metadata['merge_data'].items():
        print(f"    {rainfall}: {data['n_files']} files, {data['total_events']:,} events")


if __name__ == "__main__":
    main()
