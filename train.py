from data import CTCDataModule
from model import CRNN_CTC_Trainer

from fire import Fire

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def main(base_folder):
    data_module = CTCDataModule(base_folder)
    model = CRNN_CTC_Trainer(data_module.train_set.vocab_size(), data_module.train_set.get_i2w())
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/GrandStaff/", filename=f"Primus", 
                                   monitor="val_SER", mode='min',
                                   save_top_k=1, verbose=True)
    early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01, patience=5, mode="min", verbose=True)
    
    trainer = Trainer(max_epochs=10000, 
                      check_val_every_n_epoch=5, 
                      callbacks=[checkpointer, early_stopping])
    
    trainer.fit(model, datamodule=data_module)
    
    model.load_from_checkpoint(checkpointer.best_model_path)
    
    trainer.test(model, datamodule=data_module)
    

if __name__ == "__main__":
    Fire(main)