import code
import pickle
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import pandas as pd

trainsforms_dict = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
}


class FreeShoundTrainDataset(Dataset):
    def __init__(self, mels, labels, transform, time_mask=0.1, freq_mask=0.1, spec_aug=True):
        self.mels = mels  # (num_samples, 128, var, 3)
        self.labels = labels
        self.tfms = transform
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.spec_aug = spec_aug

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        mel = self.mels[idx]  # (128, 451, 3)
        base_dim, time_dim, _ = mel.shape
        crop = np.random.randint(0, time_dim-base_dim)  # (97)
        image = mel[:, crop: crop + base_dim, ...]  # (128, 128, 3)

        if self.spec_aug:
            freq_mask_begin = int(np.random.uniform(
                0, 1 - self.freq_mask) * base_dim)
            image[freq_mask_begin:freq_mask_begin +
                  int(self.freq_mask * base_dim), ...] = 0
            time_mask_begin = int(np.random.uniform(
                0, 1 - self.time_mask) * base_dim)
            image[:, time_mask_begin:time_mask_begin +
                  int(self.time_mask * base_dim), ...] = 0

        image = Image.fromarray(image[..., 0], mode='L')  # (128, 128)
        image = self.tfms(image).div_(255)  # (1, 128, 128)
        if self.labels is not None:
            label = np.asarray(self.labels)[idx]
            label = torch.from_numpy(label).float()
        return (image, label) if self.labels is not None else image


if __name__ == "__main__":
    path = Path('data')
    train_df = pd.read_csv(path/'train_curated.csv')
    le = LabelEncoder()
    train_df['labelcode'] = le.fit_transform(train_df['labels'])

    processed_train = pickle.load(
        open(path/'mels_train.pkl', 'rb'))  # (4970, 128, var, 3)
    y_train = train_df.labelcode.values.astype(np.long)

    train_ds = FreeShoundTrainDataset(
        processed_train, y_train, trainsforms_dict['train'])
    x, y = train_ds[0]
    print(x.shape, y)
