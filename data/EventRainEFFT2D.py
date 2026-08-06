import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class EventRainEFFT2D(Dataset):
    """
    Dataset for 2D Event-based Deraining (Frame-based).
    
    Layout expected:
    root/
      merge_data/
        1mm/*.npz
        5mm/*.npz
        ...
      raw_data/*.npz

    - Iterates through all frames in all intensity folders.
    - Returns individual 2D frames instead of sequences.
    """

    def __init__(
        self,
        root="/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/efft_results",
        purpose="train",
        allow_pickle=True,
        require_raw=True,
        realworld_frame_size=256,
    ):
        self.root = root
        self.purpose = purpose
        self.allow_pickle = allow_pickle
        self.require_raw = require_raw
        self.realworld_frame_size = int(realworld_frame_size)

        merge_root_with_purpose = os.path.join(root, "merge_data", purpose)
        merge_root_without_purpose = os.path.join(root, "merge_data")
        direct_npz_in_root = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.direct_npz_mode = False

        if os.path.isdir(merge_root_with_purpose):
            self.merge_root = merge_root_with_purpose
        elif os.path.isdir(merge_root_without_purpose):
            self.merge_root = merge_root_without_purpose
        elif len(direct_npz_in_root) > 0:
            # Real-world convenience mode: `root` itself is a single intensity folder.
            self.merge_root = root
            self.direct_npz_mode = True
        else:
            raise FileNotFoundError(
                f"Could not find merge_data folder at either "
                f"{merge_root_with_purpose} or {merge_root_without_purpose}, "
                f"and no direct npz files under {root}."
            )

        self.raw_root = os.path.join(root, "raw_data")

        # 1. Get all intensity folders
        if self.direct_npz_mode:
            single_label = os.path.basename(os.path.normpath(self.merge_root)) or "realworld"
            self.intensities = [single_label]
        else:
            self.intensities = sorted(
                [
                    d
                    for d in os.listdir(self.merge_root)
                    if os.path.isdir(os.path.join(self.merge_root, d))
                ]
            )
            if len(self.intensities) == 0:
                raise RuntimeError(f"No intensity subfolders found under {self.merge_root}")

        # 2. Get ground-truth frames (raw) if available.
        self.raw_files = sorted(glob.glob(os.path.join(self.raw_root, "*.npz")))
        self.has_raw = len(self.raw_files) > 0
        if self.require_raw and not self.has_raw:
            raise RuntimeError(
                f"raw_data was not found or empty under {self.raw_root}, "
                "but require_raw=True."
            )

        # 3. Map files per intensity
        self.merge_files_per_intensity = {}
        self.num_frames_per_intensity = []

        for mm in self.intensities:
            if self.direct_npz_mode:
                files = sorted(glob.glob(os.path.join(self.merge_root, "*.npz")))
            else:
                files = sorted(glob.glob(os.path.join(self.merge_root, mm, "*.npz")))
            
            if self.has_raw:
                # Align lengths: keep min frames between raw and this intensity.
                n_frames = min(len(files), len(self.raw_files))
            else:
                n_frames = len(files)
            
            self.merge_files_per_intensity[mm] = files[:n_frames]
            self.num_frames_per_intensity.append(n_frames)

        # 4. Total dataset length is sum of all valid frames
        self.total_frames = sum(self.num_frames_per_intensity)

    def __len__(self):
        return self.total_frames

    def _load_npz(self, path):
        """
        Loads one sample and returns complex FFT tensor.

        Supported formats:
        1) Precomputed EFFT archive containing `fft_complex`.
        2) Raw event archive containing `x, y, t, p`, converted on-the-fly.
        """
        npz = np.load(path, allow_pickle=self.allow_pickle)
        keys = set(npz.files)

        if "fft_complex" in keys:
            arr = npz["fft_complex"].astype(np.complex64)
            npz.close()
            return torch.from_numpy(arr)

        required_event_keys = {"x", "y", "p"}
        if required_event_keys.issubset(keys):
            x = npz["x"].astype(np.int32)
            y = npz["y"].astype(np.int32)
            p = npz["p"]
            npz.close()

            # Convert event stream to a fixed-size signed accumulation map.
            frame_size = self.realworld_frame_size
            x_max = int(x.max()) + 1 if x.size > 0 else 1
            y_max = int(y.max()) + 1 if y.size > 0 else 1
            x_scaled = (x.astype(np.float32) * frame_size / max(x_max, 1)).astype(np.int32)
            y_scaled = (y.astype(np.float32) * frame_size / max(y_max, 1)).astype(np.int32)
            x_scaled = np.clip(x_scaled, 0, frame_size - 1)
            y_scaled = np.clip(y_scaled, 0, frame_size - 1)

            event_map = np.zeros((frame_size, frame_size), dtype=np.float32)
            polarity = np.where(p > 0, 1.0, -1.0).astype(np.float32)
            np.add.at(event_map, (y_scaled, x_scaled), polarity)

            # Normalize to [0, 1] before FFT so inference stays numerically stable.
            min_v = float(event_map.min())
            max_v = float(event_map.max())
            if max_v > min_v:
                event_map = (event_map - min_v) / (max_v - min_v)
            else:
                event_map.fill(0.0)

            arr = np.fft.fft2(event_map).astype(np.complex64)
            return torch.from_numpy(arr)

        npz.close()
        raise KeyError(
            f"Unsupported npz format in {path}. Expected `fft_complex` or event keys {required_event_keys}."
        )

    def __getitem__(self, idx):
        """
        Returns a dictionary containing:
            - 'raw': Spatial domain image (Real), shape [H, W]
            - 'merge': Frequency domain input (Complex), shape [H, W]
            - 'rain_type': String identifier (e.g., '1mm')
        """
        # 1. Map flat idx to (intensity_index, frame_index)
        running = 0
        intensity_idx = 0
        frame_idx = 0
        
        for i, n_frames in enumerate(self.num_frames_per_intensity):
            if idx < running + n_frames:
                intensity_idx = i
                frame_idx = idx - running
                break
            running += n_frames

        mm = self.intensities[intensity_idx]

        # 2. Get file paths
        merge_path = self.merge_files_per_intensity[mm][frame_idx]

        # 3. Load data
        # Merge: Load complex (Frequency Domain)
        merge_complex = self._load_npz(merge_path)
        merge_img = torch.fft.ifft2(merge_complex).real.unsqueeze(0)

        # Raw (GT): optional for real-world inference datasets.
        if self.has_raw:
            raw_path = self.raw_files[frame_idx]
            raw_complex = self._load_npz(raw_path)
            raw_img = torch.fft.ifft2(raw_complex).real.unsqueeze(0)
        else:
            raw_img = torch.zeros_like(merge_img)

        return {
            "raw": raw_img,          # Tensor [H, W] (Real)
            "merge": merge_img,      # Tensor [H, W] (Real)
            "rain_type": mm,
            "has_raw": self.has_raw,
            "frame_name": os.path.basename(merge_path),
        }

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy.fft as fft

    # Create dataset instance
    # Ensure the root path exists on your system or update it for testing
    dataset = EventRainEFFT2D(purpose="train") 
    print(f"Dataset length (Total Frames): {len(dataset)}")

    if len(dataset) > 0:
        sample_idx = 0
        sample = dataset[sample_idx]
        
        print(f"\nSample at idx={sample_idx}:")
        print(f"  Rain type: {sample['rain_type']}")
        print(f"  Raw Shape: {sample['raw'].shape} (dtype: {sample['raw'].dtype})")
        print(f"  Merge Shape: {sample['merge'].shape} (dtype: {sample['merge'].dtype})")

        # Visualization
        # Raw is already spatial/real
        raw_img = sample['raw'].numpy()
        
        # Merge is complex, so we IFFT it for visualization
        merge_img = sample['merge'].numpy()

        print("Range of Raw Image: ", raw_img.min(), raw_img.max())
        print("Range of Merge Image: ", merge_img.min(), merge_img.max())

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(raw_img, cmap='gray')
        plt.title(f"Target (Raw)\nSpatial Real")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(merge_img, cmap='gray')
        plt.title(f"Input (Merge)\nSpatial Real")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig('check_2d_dataset.png')
        print("\nSaved visualization to check_2d_dataset.png")
        plt.close()