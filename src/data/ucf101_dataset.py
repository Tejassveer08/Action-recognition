import torch
from torch.utils.data import Dataset
import torchvision.io as io

class UCF101Dataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video, _, _ = io.read_video(video_path, pts_unit='sec')
        video = video.permute(0, 3, 1, 2)  # T, C, H, W
        if self.transform:
            video = self.transform(video)
        return video.float() / 255.0, label
