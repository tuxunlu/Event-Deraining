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
    ):
        self.root = root
        self.purpose = purpose
        self.allow_pickle = allow_pickle

        self.merge_root = os.path.join(root, "merge_data", purpose)
        self.raw_root = os.path.join(root, "raw_data")

        # 1. Get all intensity folders
        self.intensities = sorted(
            [
                d
                for d in os.listdir(self.merge_root)
                if os.path.isdir(os.path.join(self.merge_root, d))
            ]
        )

        # 2. Get ground-truth frames (raw)
        self.raw_files = sorted(glob.glob(os.path.join(self.raw_root, "*.npz")))

        # 3. Map files per intensity
        self.merge_files_per_intensity = {}
        self.num_frames_per_intensity = []

        for mm in self.intensities:
            files = sorted(glob.glob(os.path.join(self.merge_root, mm, "*.npz")))
            
            # Align lengths: keep min frames between raw and this intensity
            n_frames = min(len(files), len(self.raw_files))
            
            self.merge_files_per_intensity[mm] = files[:n_frames]
            self.num_frames_per_intensity.append(n_frames)

        # 4. Total dataset length is sum of all valid frames
        self.total_frames = sum(self.num_frames_per_intensity)

    def __len__(self):
        return self.total_frames

    def _load_npz(self, path):
        # Loads complex FFT data
        arr = np.load(path, allow_pickle=self.allow_pickle)["fft_complex"].astype(np.complex64)
        return torch.from_numpy(arr)

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
        raw_path = self.raw_files[frame_idx]
        merge_path = self.merge_files_per_intensity[mm][frame_idx]

        # 3. Load data
        # Raw: Load complex -> IFFT -> Real (Spatial Domain)
        raw_complex = self._load_npz(raw_path)
        raw_img = torch.fft.ifft2(raw_complex).real.unsqueeze(0)

        # Merge: Load complex (Frequency Domain)
        merge_complex = self._load_npz(merge_path)
        merge_img = torch.fft.ifft2(merge_complex).real.unsqueeze(0)

        return {
            "raw": raw_img,          # Tensor [H, W] (Real)
            "merge": merge_img,      # Tensor [H, W] (Real)
            "rain_type": mm
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