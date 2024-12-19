import os
import torch
import numpy as np
from rich import progress
from torchvision import transforms
from lightning import LightningDataModule
import cv2
import json
from torch.utils.data import Dataset
from utils import check_and_retrieveVocabulary

FIXED_HEIGHT = 128

def convert_img_to_tensor(image, force_one_channel=False):
    transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.RandomInvert(p=1),
        transforms.ToTensor()]
    )

    image = transform(image)

    return image

def load_set(partition_file, base_path):
    x = []
    y = []
    
    with open(partition_file, 'r') as f:
        filepaths = f.readlines()
        for file in progress.track(filepaths):
            file = file.replace('\n', '')
            try:
                with open(file, 'r') as json_file:
                    json_data = json.load(json_file)
                    symbols = json_data['pages'][0]['regions'][0]['symbols']
                    transcription = [f"{symbol['agnostic_symbol_type']}:{symbol['position_in_staff']}" for symbol in symbols]
                    image = cv2.imread(base_path + json_data["path"], cv2.IMREAD_GRAYSCALE)
                    width = int(float(FIXED_HEIGHT * image.shape[1]) / image.shape[0])
                    image = cv2.resize(image, (width, FIXED_HEIGHT))
                    x.append(image)
                    y.append(transcription)
                        
            except Exception as e:
                print('Error loading file:', file)
                
    return x, y

def batch_preparation_ctc(data):
    images = [sample[0] for sample in data]
    gt = [sample[1] for sample in data]
    L = [sample[2] for sample in data]
    T = [sample[3] for sample in data]

    max_image_width = max([img.shape[2] for img in images])
    max_image_height = max([img.shape[1] for img in images])

    X_train = torch.zeros(size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32)

    for i, img in enumerate(images):
        c, h, w = img.size()
        X_train[i, :, :h, :w] = img
    
    max_length_seq = max([len(w) for w in gt])
    Y_train = torch.zeros(size=[len(gt),max_length_seq])
    for i, seq in enumerate(gt):
        Y_train[i, 0:len(seq)] = torch.from_numpy(np.asarray([char for char in seq]))

    return X_train, Y_train, L, T

class CTCDataset(Dataset):
    def __init__(self, setfile, base_path) -> None:
        self.x, self.y = load_set(setfile, base_path)
        self.w2i, self.i2w = None, None
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        image = convert_img_to_tensor(self.x[index])
        gt = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        
        return image, gt, image.shape[2] // 2, len(gt)
    
    def get_max_hw(self):
        m_width = np.max([img.shape[1] for img in self.x])
        m_height = np.max([img.shape[0] for img in self.x])

        return m_width, m_height

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y
    
    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
    
    def get_i2w(self):
        return self.i2w

class CTCDataModule(LightningDataModule):
    def __init__(self, base_path) -> None:
        super().__init__()
        self.vocab_name = "CTCPrimus"
        self.batch_size = 16
        self.num_workers = 25
        self.train_set = CTCDataset(setfile="train.txt", base_path=base_path)
        self.val_set = CTCDataset(setfile="val.txt", base_path=base_path)
        self.test_set = CTCDataset(setfile="test.txt", base_path=base_path)

        w2i, i2w = check_and_retrieveVocabulary([self.train_set.get_gt(), self.val_set.get_gt(), self.test_set.get_gt()], "vocab/", f"{self.vocab_name}")
        
        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=batch_preparation_ctc)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=1, num_workers=self.num_workers, collate_fn=batch_preparation_ctc)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=1, num_workers=self.num_workers, collate_fn=batch_preparation_ctc)



if __name__ == "__main__":
    data = load_set('train.txt')

