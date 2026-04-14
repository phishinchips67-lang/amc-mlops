
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

MODULATIONS = ['8PSK','AM-DSB','AM-SSB','BPSK','CPFSK',
                'GFSK','PAM4','QAM16','QAM64','QPSK','WBFM']
MOD_TO_IDX = {m: i for i, m in enumerate(MODULATIONS)}
ALL_SNRS = list(range(-20, 19, 2))

class RadioMLDataset(Dataset):
    def __init__(self, data_dict, snr_filter=None):
        self.samples = []
        self.labels = []
        self.snrs = []
        for (mod, snr), iq_array in data_dict.items():
            if snr_filter is not None and snr not in snr_filter:
                continue
            for iq in iq_array:
                self.samples.append(iq.astype(np.float32))
                self.labels.append(MOD_TO_IDX[mod])
                self.snrs.append(snr)
        self.samples = np.array(self.samples)
        self.labels  = np.array(self.labels)
        self.snrs    = np.array(self.snrs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx])
        # Normalize per sample: zero mean, unit variance
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def load_data(pkl_path="data/RML2016.10a_dict.pkl"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data

def get_dataloaders(pkl_path="data/RML2016.10a_dict.pkl",
                    snr_filter=None, batch_size=256, val_split=0.2):
    data = load_data(pkl_path)
    dataset = RadioMLDataset(data, snr_filter=snr_filter)
    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, dataset
