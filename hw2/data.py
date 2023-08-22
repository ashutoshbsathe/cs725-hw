import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Basic data loaders for both the datasets
# You should not need to modify this at all

class DummyNumPyDataset(Dataset):
    def __init__(self, x_path, y_path):
        super().__init__()
        self.x = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LitNumPyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def setup(self, stage):
        if stage == 'fit':
            self.trainset = DummyNumPyDataset(
                self.data_dir + '/train_x.npy',
                self.data_dir + '/train_y.npy',
            )
            self.validset = DummyNumPyDataset(
                self.data_dir + '/valid_x.npy',
                self.data_dir + '/valid_y.npy',
            )
        
        if stage == 'validate' or stage == 'predict':
            self.validset = DummyNumPyDataset(
                self.data_dir + '/valid_x.npy',
                self.data_dir + '/valid_y.npy',
            )

        if stage == 'test':
            self.testset = DummyNumPyDataset(
                self.data_dir + '/test_x.npy',
                self.data_dir + '/test_y.npy',
            )

    def train_dataloader(self):
        return DataLoader(self.trainset, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validset, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.testset, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.validset, self.batch_size)

class LitSimpleDataModule(LitNumPyDataModule):
    def __init__(self, batch_size=64):
        super().__init__(data_dir='./data/simple/', batch_size=batch_size)

class LitDigitsDataModule(LitNumPyDataModule):
    def __init__(self, batch_size=64):
        super().__init__(data_dir='./data/digits/', batch_size=batch_size)