import torch
from itertools import groupby
import torch.nn as nn
from torchinfo import summary
import lightning.pytorch as L
from utils import levenshtein

class CRNN(nn.Module):
    def __init__(self, output_projection):
        super(CRNN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding='same'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
        )
        
        self.rnn_features = (128 // (2 ** 4)) * 128
        self.rnn = nn.LSTM(
            input_size=self.rnn_features,
            hidden_size=256,
            num_layers=2,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.25)
        self.vocab_projection = nn.Linear(512, output_projection)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], self.rnn_features)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.vocab_projection(x)
        x = self.softmax(x)
        x = x.permute(1, 0, 2).contiguous()
        
        return x


class CRNN_CTC_Trainer(L.LightningModule):
    def __init__(self, output_size, i2w):
        super().__init__()
        self.model = CRNN(output_size+1)
        self.loss = nn.CTCLoss(blank=output_size)
        self.i2w = i2w
        self.save_hyperparameters(ignore=['model'])
        self.summary()
        self.blank_idx = output_size
        
        self.dec_val_ex = []
        self.gt_val_ex = []
        
    def summary(self):
        summary(self.model, input_size=(8, 1, 128, 512))
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def training_step(self, batch):
        X_tr, Y_tr, L_tr, T_tr = batch
        predictions = self.forward(X_tr)
        loss = self.loss(predictions, Y_tr, L_tr, T_tr)
        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def compute_prediction(self, batch):
        X, Y, _, _ = batch
        pred = self.forward(X)
        pred = pred.permute(1,0,2).contiguous()
        pred = pred[0]
        out_best = torch.argmax(pred,dim=1)
        out_best = [k for k, g in groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c.item() != self.blank_idx:
                decoded.append(c.item())
        
        decoded = [self.i2w[tok] for tok in decoded]
        gt = [self.i2w[int(tok.item())] for tok in Y[0]]

        return decoded, gt

    def validation_step(self, val_batch, batch_idx):
        dec, gt = self.compute_prediction(val_batch)

        self.dec_val_ex.append(dec)
        self.gt_val_ex.append(gt)
    
    def test_step(self, val_batch, batch_idx):
        dec, gt = self.compute_prediction(val_batch)

        self.dec_val_ex.append(dec)
        self.gt_val_ex.append(gt)
    
    def on_validation_epoch_end(self, type='val'):
        ed = 0
        length = 0
        for dec, gt in zip(self.dec_val_ex, self.gt_val_ex):
            ed += levenshtein(dec, gt)
            length += len(gt)
        
        ser = (ed / length) * 100
        self.log(f'{type}_SER', ser, on_epoch=True, prog_bar=True)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(type='test')

if __name__ == "__main__":
    model = CRNN_CTC_Trainer(100, {})
        
        
    