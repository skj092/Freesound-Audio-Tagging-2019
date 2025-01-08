from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from torchvision import transforms
from dataset import FreeShoundTrainDataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
import torch
from models import Model, Classifier
from sklearn.metrics import accuracy_score
from fastprogress.fastprogress import master_bar, progress_bar
import os
from tqdm import tqdm
from torch.nn.functional import softmax

batch_size = 8
lr = 1e-3
tmax = 10
eta_min = 1e-5
num_epochs = 10
num_classes = 80
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


def make_prediction(model, test_dl):
    model.eval()
    preds = []
    for xb in tqdm(test_dl):
        with torch.no_grad():
            logits = model(xb.to(device))
            pred = softmax(logits, dim=1)
        preds.extend(pred.cpu().numpy())
    return preds


def create_unique_labels(all_labels):
    label_dict = {}
    all_labels_set = []
    first_labels_set = []
    for labs in all_labels:
        lab = labs.split(',')
        for l in lab:
            if l in label_dict:
                label_dict[l] = label_dict[l] + 1
            else:
                label_dict[l] = 0

        all_labels_set.append(set(lab))
        first_labels_set.append(lab[0])
    classes = list(label_dict.keys())

    return label_dict, classes, all_labels_set, first_labels_set


if __name__ == "__main__":
    # load input data
    path = Path('data')
    train_df = pd.read_csv(path/'train_curated.csv')
    train_df = transform_labels(train_df)
    y_train = train_df.iloc[:, 1:]

    test_images = sorted(os.listdir(path/'test_files'))
    sub = pd.read_csv(path/'sample_submission.csv')

    # load preprocessed audios and corresponding labels
    processed_train = pickle.load(
        open(path/'mels_train.pkl', 'rb'))  # (4970, 128, var, 3)
    processed_test = pickle.load(
        open(path/'mels_test.pkl', 'rb'))  # (4970, 128, var, 3)
    y_train = train_df.iloc[:, 1:]

    # create custom dataset and dataloader
    train_ds = FreeShoundTrainDataset(
        processed_train, y_train, trainsforms_dict['train'])
    if debug:
        train_ds = Subset(train_ds, range(100))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = FreeShoundTrainDataset(
        processed_test, None, trainsforms_dict['train'])
    if debug:
        test_ds = Subset(test_ds, range(100))
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print('=======Data Loader Prepared=========')

    # load model
    model = Classifier(num_classes=80)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.to(device)
    print('=======Model Loaded=========')

    preds = make_prediction(model, test_dl)
    sub = pd.DataFrame(preds, columns=y_train.columns)
    if debug:
        test_fns = sorted(os.listdir(path/'test_files'))[:100]
    else:
        test_fns = sorted(os.listdir(path/'test_files'))
    sub['fname'] = test_fns
    sub = sub[['fname'] + [col for col in sub.columns if col != 'fname']]
    import code
    code.interact(local=locals())
