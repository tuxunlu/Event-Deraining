#!/usr/bin/env python3
"""Visualize EFFT Results"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path


class EFFTVisualizer:

    def __init__(self,
                 data_root="data/eventrain_KITTI/synthetic/synthetic_KITTI/synthetic",
                 efft_root="efft_results"):
        self.data_root = data_root
        self.efft_root = efft_root
        self.frame_size = 256
        self.orig_width = 460
        self.orig_height = 352

    def downsample_coordinates(self, x, y):
        x_scaled = (x * self.frame_size / self.orig_width).astype(np.int32)
        y_scaled = (y * self.frame_size / self.orig_height).astype(np.int32)
        x_scaled = np.clip(x_scaled, 0, self.frame_size - 1)
        y_scaled = np.clip(y_scaled, 0, self.frame_size - 1)
        return x_scaled, y_scaled

    def create_event_frame(self, npz_file):
        data = np.load(npz_file)
        x = data['x']
        y = data['y']
        p = data['p']
        data.close()

        x_scaled, y_scaled = self.downsample_coordinates(x, y)

        frame_on = np.zeros((self.frame_size, self.frame_size))
        frame_off = np.zeros((self.frame_size, self.frame_size))

        for i in range(len(x)):
            if p[i] == 1:
                frame_on[y_scaled[i], x_scaled[i]] += 1
            else:
                frame_off[y_scaled[i], x_scaled[i]] += 1

        return frame_on, frame_off

    def visualize_single_window(self, rainfall_mm, time_window_idx, save_path=None):
        event_file = os.path.join(
            self.data_root, "merge_data", f"{rainfall_mm}mm",
            f"{str(time_window_idx).zfill(10)}.npz"
        )

        fft_file = os.path.join(
            self.efft_root, "merge_data", f"{rainfall_mm}mm",
            f"{str(time_window_idx).zfill(10)}.npz"
        )

        if not os.path.exists(event_file):
            print(f"Event file not found: {event_file}")
            return

        if not os.path.exists(fft_file):
            print(f"FFT file not found: {fft_file}")
            return

        frame_on, frame_off = self.create_event_frame(event_file)
        fft_data = np.load(fft_file)

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        event_data = np.load(event_file)
        x = event_data['x']
        y = event_data['y']
        t = event_data['t']
        p = event_data['p']

        fig.suptitle(
            f'EFFT Results: Rainfall {rainfall_mm}mm, Time Window {time_window_idx}\n'
            f'Total Events: {len(x):,} | Duration: {(t.max()-t.min())/1000:.1f}ms',
            fontsize=14, fontweight='bold'
        )

        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(np.log10(frame_on + 1), cmap='Reds', origin='upper')
        ax1.set_title('ON Events (log scale)')
        ax1.set_xlabel('X (col)')
        ax1.set_ylabel('Y (row)')
        plt.colorbar(im1, ax=ax1, label='log10(count+1)')

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(np.log10(frame_off + 1), cmap='Blues', origin='upper')
        ax2.set_title('OFF Events (log scale)')
        ax2.set_xlabel('X (col)')
        ax2.set_ylabel('Y (row)')
        plt.colorbar(im2, ax=ax2, label='log10(count+1)')

        ax3 = fig.add_subplot(gs[0, 2])
        combined = frame_on - frame_off
        im3 = ax3.imshow(combined, cmap='RdBu_r', origin='upper')
        ax3.set_title('ON - OFF Events')
        ax3.set_xlabel('X (col)')
        ax3.set_ylabel('Y (row)')
        plt.colorbar(im3, ax=ax3, label='ON - OFF')

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        stats_text = f"""
Events: {len(x):,}
ON: {np.sum(p==1):,}
OFF: {np.sum(p==-1):,}

Time: {(t.max()-t.min())/1000:.2f}ms
Range: [{x.min()}, {x.max()}] × [{y.min()}, {y.max()}]
Downsampled: {self.frame_size}×{self.frame_size}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')

        magnitude = fft_data['fft_magnitude']
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(np.log10(magnitude + 1), cmap='viridis', origin='upper')
        ax5.set_title('FFT Magnitude (log scale)')
        ax5.set_xlabel('Frequency X')
        ax5.set_ylabel('Frequency Y')
        plt.colorbar(im5, ax=ax5, label='log10(magnitude+1)')

        magnitude_shifted = np.fft.fftshift(magnitude)
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(np.log10(magnitude_shifted + 1), cmap='hot', origin='upper')
        ax6.set_title('FFT Magnitude (centered)')
        ax6.set_xlabel('Frequency X')
        ax6.set_ylabel('Frequency Y')
        plt.colorbar(im6, ax=ax6, label='log10(magnitude+1)')

        phase = fft_data['fft_phase']
        ax7 = fig.add_subplot(gs[1, 2])
        im7 = ax7.imshow(phase, cmap='twilight', origin='upper', vmin=-np.pi, vmax=np.pi)
        ax7.set_title('FFT Phase')
        ax7.set_xlabel('Frequency X')
        ax7.set_ylabel('Frequency Y')
        plt.colorbar(im7, ax=ax7, label='Phase (radians)')

        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis('off')
        fft_stats = f"""
Magnitude:
  Min:  {magnitude.min():.2f}
  Max:  {magnitude.max():.2f}
  Mean: {magnitude.mean():.2f}

DC: {magnitude[0,0]:.2f}
        """
        ax8.text(0.1, 0.5, fft_stats, fontsize=10, family='monospace',
                verticalalignment='center')

        ax9 = fig.add_subplot(gs[2, 0])
        mag_horizontal = magnitude_shifted[self.frame_size//2, :]
        freq_horizontal = np.fft.fftshift(np.fft.fftfreq(self.frame_size))
        ax9.plot(freq_horizontal, mag_horizontal, 'b-', linewidth=1)
        ax9.set_title('Horizontal Frequency Profile')
        ax9.set_xlabel('Normalized Frequency')
        ax9.set_ylabel('Magnitude')
        ax9.grid(True, alpha=0.3)
        ax9.set_xlim(-0.5, 0.5)

        ax10 = fig.add_subplot(gs[2, 1])
        mag_vertical = magnitude_shifted[:, self.frame_size//2]
        freq_vertical = np.fft.fftshift(np.fft.fftfreq(self.frame_size))
        ax10.plot(freq_vertical, mag_vertical, 'r-', linewidth=1)
        ax10.set_title('Vertical Frequency Profile')
        ax10.set_xlabel('Normalized Frequency')
        ax10.set_ylabel('Magnitude')
        ax10.grid(True, alpha=0.3)
        ax10.set_xlim(-0.5, 0.5)

        ax11 = fig.add_subplot(gs[2, 2])
        y_grid, x_grid = np.ogrid[:self.frame_size, :self.frame_size]
        center = self.frame_size // 2
        r = np.sqrt((x_grid - center)**2 + (y_grid - center)**2).astype(int)
        r = r.flatten()
        mag_flat = magnitude_shifted.flatten()

        r_max = center
        radial_profile = np.zeros(r_max)
        for i in range(r_max):
            mask = (r >= i) & (r < i+1)
            if np.any(mask):
                radial_profile[i] = np.mean(mag_flat[mask])

        ax11.plot(np.arange(r_max), radial_profile, 'g-', linewidth=2)
        ax11.set_title('Radial Frequency Profile')
        ax11.set_xlabel('Frequency (pixels from center)')
        ax11.set_ylabel('Average Magnitude')
        ax11.grid(True, alpha=0.3)

        ax12 = fig.add_subplot(gs[2, 3])
        real = fft_data['fft_real'].flatten()
        imag = fft_data['fft_imag'].flatten()
        subsample = max(1, len(real) // 10000)
        ax12.scatter(real[::subsample], imag[::subsample], s=1, alpha=0.3, c='purple')
        ax12.set_title('Complex Plane')
        ax12.set_xlabel('Real')
        ax12.set_ylabel('Imaginary')
        ax12.grid(True, alpha=0.3)
        ax12.axhline(y=0, color='k', linewidth=0.5)
        ax12.axvline(x=0, color='k', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")

        plt.show()

        event_data.close()
        fft_data.close()

    def compare_rainfall_intensities(self, time_window_idx=0, save_path=None):
        merge_dir = os.path.join(self.efft_root, "merge_data")
        rainfall_dirs = sorted([d.name for d in Path(merge_dir).iterdir() if d.is_dir()])
        rainfall_intensities = [int(d.replace('mm', '')) for d in rainfall_dirs]

        selected = [1, 10, 25, 50, 75, 100, 150, 200]
        selected = [r for r in selected if r in rainfall_intensities]

        n_rainfalls = len(selected)
        fig, axes = plt.subplots(2, n_rainfalls, figsize=(3*n_rainfalls, 6))

        fig.suptitle(f'FFT Comparison Across Rainfall Intensities (Time Window {time_window_idx})',
                     fontsize=14, fontweight='bold')

        for idx, rainfall in enumerate(selected):
            fft_file = os.path.join(
                self.efft_root, "merge_data", f"{rainfall}mm",
                f"{str(time_window_idx).zfill(10)}.npz"
            )

            if not os.path.exists(fft_file):
                print(f"FFT file not found: {fft_file}")
                continue

            fft_data = np.load(fft_file)
            magnitude = fft_data['fft_magnitude']
            magnitude_shifted = np.fft.fftshift(magnitude)

            ax1 = axes[0, idx] if n_rainfalls > 1 else axes[0]
            im1 = ax1.imshow(np.log10(magnitude_shifted + 1), cmap='hot', origin='upper')
            ax1.set_title(f'{rainfall}mm')
            ax1.set_xticks([])
            ax1.set_yticks([])
            if idx == 0:
                ax1.set_ylabel('Magnitude (log)')

            ax2 = axes[1, idx] if n_rainfalls > 1 else axes[1]
            y_grid, x_grid = np.ogrid[:self.frame_size, :self.frame_size]
            center = self.frame_size // 2
            r = np.sqrt((x_grid - center)**2 + (y_grid - center)**2).astype(int)
            r = r.flatten()
            mag_flat = magnitude_shifted.flatten()

            r_max = center
            radial_profile = np.zeros(r_max)
            for i in range(r_max):
                mask = (r >= i) & (r < i+1)
                if np.any(mask):
                    radial_profile[i] = np.mean(mag_flat[mask])

            ax2.plot(np.arange(r_max), radial_profile, linewidth=2)
            ax2.set_xlabel('Frequency')
            if idx == 0:
                ax2.set_ylabel('Avg Magnitude')
            ax2.grid(True, alpha=0.3)

            fft_data.close()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")

        plt.show()

    def compare_clean_vs_rainy(self, rainfall_mm=50, time_window_idx=0, save_path=None):
        raw_fft_file = os.path.join(
            self.efft_root, "raw_data",
            f"{str(time_window_idx).zfill(10)}.npz"
        )

        merge_fft_file = os.path.join(
            self.efft_root, "merge_data", f"{rainfall_mm}mm",
            f"{str(time_window_idx).zfill(10)}.npz"
        )

        if not os.path.exists(raw_fft_file) or not os.path.exists(merge_fft_file):
            print("FFT files not found")
            return

        raw_fft = np.load(raw_fft_file)
        merge_fft = np.load(merge_fft_file)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        fig.suptitle(
            f'Clean vs Rainy FFT Comparison\n'
            f'Rainfall: {rainfall_mm}mm | Time Window: {time_window_idx}',
            fontsize=14, fontweight='bold'
        )

        raw_mag = raw_fft['fft_magnitude']
        raw_mag_shifted = np.fft.fftshift(raw_mag)

        ax1 = axes[0, 0]
        im1 = ax1.imshow(np.log10(raw_mag_shifted + 1), cmap='viridis', origin='upper')
        ax1.set_title('Clean Events - FFT Magnitude')
        plt.colorbar(im1, ax=ax1)

        ax2 = axes[0, 1]
        im2 = ax2.imshow(raw_fft['fft_phase'], cmap='twilight', origin='upper')
        ax2.set_title('Clean Events - FFT Phase')
        plt.colorbar(im2, ax=ax2)

        ax3 = axes[0, 2]
        y_grid, x_grid = np.ogrid[:self.frame_size, :self.frame_size]
        center = self.frame_size // 2
        r = np.sqrt((x_grid - center)**2 + (y_grid - center)**2).astype(int)
        r = r.flatten()
        mag_flat = raw_mag_shifted.flatten()
        r_max = center
        radial_profile_clean = np.zeros(r_max)
        for i in range(r_max):
            mask = (r >= i) & (r < i+1)
            if np.any(mask):
                radial_profile_clean[i] = np.mean(mag_flat[mask])
        ax3.plot(np.arange(r_max), radial_profile_clean, 'b-', linewidth=2, label='Clean')

        merge_mag = merge_fft['fft_magnitude']
        merge_mag_shifted = np.fft.fftshift(merge_mag)

        ax4 = axes[1, 0]
        im4 = ax4.imshow(np.log10(merge_mag_shifted + 1), cmap='viridis', origin='upper')
        ax4.set_title(f'Rainy Events ({rainfall_mm}mm) - FFT Magnitude')
        plt.colorbar(im4, ax=ax4)

        ax5 = axes[1, 1]
        im5 = ax5.imshow(merge_fft['fft_phase'], cmap='twilight', origin='upper')
        ax5.set_title(f'Rainy Events ({rainfall_mm}mm) - FFT Phase')
        plt.colorbar(im5, ax=ax5)

        ax6 = axes[1, 2]
        mag_flat_merge = merge_mag_shifted.flatten()
        radial_profile_rainy = np.zeros(r_max)
        for i in range(r_max):
            mask = (r >= i) & (r < i+1)
            if np.any(mask):
                radial_profile_rainy[i] = np.mean(mag_flat_merge[mask])
        ax6.plot(np.arange(r_max), radial_profile_rainy, 'r-', linewidth=2, label='Rainy')

        ax3.plot(np.arange(r_max), radial_profile_rainy, 'r-', linewidth=2, label='Rainy', alpha=0.7)
        ax3.set_title('Radial Profile Comparison')
        ax3.set_xlabel('Frequency (from center)')
        ax3.set_ylabel('Average Magnitude')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax6.plot(np.arange(r_max), radial_profile_clean, 'b-', linewidth=2, label='Clean', alpha=0.7)
        ax6.set_title('Radial Profile (both)')
        ax6.set_xlabel('Frequency (from center)')
        ax6.set_ylabel('Average Magnitude')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")

        plt.show()

        raw_fft.close()
        merge_fft.close()


def main():
    print("\nEFFT Results Visualization")

    visualizer = EFFTVisualizer()

    print("\nAvailable visualizations:")
    print("  1. Single time window (detailed)")
    print("  2. Compare rainfall intensities")
    print("  3. Compare clean vs rainy")
    print("  4. All of the above")

    choice = input("\nSelect visualization (1-4, default=1): ").strip()
    choice = choice if choice else "1"

    if choice in ["1", "4"]:
        print("\nVisualization 1: Single Time Window")
        visualizer.visualize_single_window(
            rainfall_mm=50,
            time_window_idx=0,
            save_path="efft_vis_single_window.png"
        )

    if choice in ["2", "4"]:
        print("\nVisualization 2: Compare Rainfall Intensities")
        visualizer.compare_rainfall_intensities(
            time_window_idx=0,
            save_path="efft_vis_rainfall_comparison.png"
        )

    if choice in ["3", "4"]:
        print("\nVisualization 3: Clean vs Rainy")
        visualizer.compare_clean_vs_rainy(
            rainfall_mm=50,
            time_window_idx=0,
            save_path="efft_vis_clean_vs_rainy.png"
        )

    print("\nVisualization complete")


if __name__ == "__main__":
    main()
