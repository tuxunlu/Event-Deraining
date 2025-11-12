import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class EventRainEFFT(Dataset):
    """
    Layout expected:

    root/
      merge_data/
        1mm/*.npz
        5mm/*.npz
        ...
      raw_data/*.npz

    - Each intensity folder has the same structure as raw_data (same filenames).
    - We slice EACH intensity into non-overlapping sequences of length `seq_len`.
    - Indexing is flattened over (intensity, sequence_idx).

    Example:
        2 intensities: [1mm, 5mm]
        each has 10 frames
        seq_len = 5

        num_seq_per_intensity = 10 // 5 = 2
        __len__ = 2 intensities * 2 seqs = 4

        idx=0 -> 1mm, frames [0..4]
        idx=1 -> 1mm, frames [5..9]
        idx=2 -> 5mm, frames [0..4]
        idx=3 -> 5mm, frames [5..9]
    """

    def __init__(
        self,
        root="/fs/nexus-scratch/tuxunlu/git/event-based-deraining/dataset/efft_results",
        seq_len=5,
        purpose="train",
        allow_pickle=True,
    ):
        self.root = root
        self.seq_len = seq_len
        self.purpose = purpose
        self.allow_pickle = allow_pickle

        self.merge_root = os.path.join(root, "merge_data", purpose)
        self.raw_root = os.path.join(root, "raw_data")

        # all intensity folders under merge_data
        self.intensities = sorted(
            [
                d
                for d in os.listdir(self.merge_root)
                if os.path.isdir(os.path.join(self.merge_root, d))
            ]
        )

        # ground-truth frames
        self.raw_files = sorted(glob.glob(os.path.join(self.raw_root, "*.npz")))

        # for each intensity, list its frames
        self.merge_files_per_intensity = {}
        self.num_frames_per_intensity = []
        for mm in self.intensities:
            files = sorted(glob.glob(os.path.join(self.merge_root, mm, "*.npz")))
            # keep min with raw in case one side is shorter
            n_frames = min(len(files), len(self.raw_files))
            self.merge_files_per_intensity[mm] = files
            self.num_frames_per_intensity.append(n_frames)

        # assume all intensities should use their own min count
        # but seq_len must fit -> num_seq_per_intensity is per-intensity
        self.num_seq_per_intensity = []
        for n_frames in self.num_frames_per_intensity:
            self.num_seq_per_intensity.append(n_frames // self.seq_len)

        # total dataset length = sum over intensities
        self.total_seqs = sum(self.num_seq_per_intensity)

    def __len__(self):
        return self.total_seqs

    def _load_npz(self, path):
        arr = np.load(path, allow_pickle=False)["fft_complex"].astype(np.complex64)
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        """
        Flattened mapping:
        idx walks through intensities in order.

        Suppose:
            intensity 0 has S0 sequences
            intensity 1 has S1 sequences
        Then:
            idx in [0, S0-1]         -> intensity 0
            idx in [S0, S0+S1-1]     -> intensity 1
            ...
        """
        # figure out which intensity this idx belongs to
        running = 0
        intensity_idx = None
        seq_idx_in_intensity = None
        for i, n_seq in enumerate(self.num_seq_per_intensity):
            if idx < running + n_seq:
                intensity_idx = i
                seq_idx_in_intensity = idx - running
                break
            running += n_seq

        mm = self.intensities[intensity_idx]

        # frame range for this sequence inside this intensity
        start = seq_idx_in_intensity * self.seq_len
        end = start + self.seq_len

        # load raw frames (use same indices)
        raw_seq = torch.stack([self._load_npz(p) for p in self.raw_files[start:end]])

        # load rainy frames from this particular intensity
        merge_seq = torch.stack([self._load_npz(p) for p in self.merge_files_per_intensity[mm][start:end]])

        return {
            "raw": raw_seq,                   # list length = seq_len
            "merge": merge_seq,               # list length = seq_len (same intensity)
        }

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy.fft as fft
    dataset = EventRainEFFT(seq_len=5, purpose="test")
    print(f"Dataset length: {len(dataset)}")

    sample_idx = 10
    sample = dataset[sample_idx]
    print(f"Sample at idx={sample_idx}:")
    print(f"  Rain type: {sample['rain_type']}")
    print(f"  Number of raw frames: {len(sample['raw'])}")
    print(f"  Number of merge frames: {len(sample['merge'])}")

    # visualize ifft of the first frame in raw
    first_raw = sample['raw'][0]
    fft_data = first_raw['fft_complex']
    ifft_result = fft.ifft2(fft_data)
    ifft_img = ifft_result.real
    plt.imshow(ifft_img, cmap='gray')
    plt.title('Inverse FFT of First Raw Frame')
    plt.colorbar()
    plt.savefig('ifft_first_raw_frame.png')
    plt.close()

    # visualize ifft of the first frame in merge
    first_merge = sample['merge'][0]
    fft_data_merge = first_merge['fft_complex']
    ifft_result_merge = fft.ifft2(fft_data_merge)
    ifft_img_merge = ifft_result_merge.real
    plt.imshow(ifft_img_merge, cmap='gray')
    plt.title('Inverse FFT of First Merge Frame')  
    plt.colorbar()
    plt.savefig('ifft_first_merge_frame.png')
    plt.close()
