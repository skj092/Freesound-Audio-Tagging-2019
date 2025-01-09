from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from torchvision import transforms
from dataset import FreeShoundTrainDataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset, random_split
from torch import nn, optim
import torch
from models import Model, Classifier
from sklearn.metrics import accuracy_score
from fastprogress.fastprogress import master_bar, progress_bar
from sklearn.model_selection import train_test_split
from engine import train_model

batch_size = 8
lr = 1e-3
tmax = 10
eta_min = 1e-5
num_epochs = 10
debug = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def transform_labels(df):
    df['labels'] = df['labels'].str.split(',')
    unique_labels = set()
    for labels in df['labels']:
        unique_labels.update(labels)
    unique_labels = sorted(list(unique_labels))

    for label in unique_labels:
        df[label] = df['labels'].apply(lambda x: 1 if label in x else 0)

    df = df.drop('labels', axis=1)

    return df


if __name__ == "__main__":
    # load input data
    path = Path('data')
    train_df = pd.read_csv(path/'train_curated.csv')
    train_df = transform_labels(train_df)

    # load preprocessed audios and corresponding labels
    processed_train = pickle.load(
        open(path/'mels_train.pkl', 'rb'))  # (4970, 128, var, 3)
    y_train = train_df.iloc[:, 1:]

    # create custom dataset and dataloader
    ds = FreeShoundTrainDataset(
        processed_train, y_train, trainsforms_dict['train'])
    if debug:
        ds = Subset(ds, range(100))
    train_size = int(len(ds) * 0.8)
    valid_size = len(ds) - train_size
    train_ds, valid_ds = random_split(ds, [train_size, valid_size])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # load model
    # model = Model(num_classes=len(le.classes_), pretrained=True)
    model = Classifier(num_classes=80)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, amsgrad=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tmax, eta_min=eta_min)

    settings = {
        'model': model,
        'train_dl': train_dl,
        'valid_dl': valid_dl,
        'loss_fn': loss_fn,
        'num_epochs': num_epochs,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'device': device
    }
    train_model(**settings)


    torch.save(model.state_dict(), 'model.pt')
