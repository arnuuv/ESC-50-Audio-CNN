import sys
import modal
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from pathlib import Path
import torchaudio
import torch
import sys
import torch.nn as nn
import torchaudio.transforms as T
from model import AudioCNN
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

app = modal.App("audio-cnn")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
    .run_commands([
        "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
        "cd /tmp && unzip esc50.zip",
        "mkdir -p /opt/esc50-data",
        "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
        "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
    ])
    .add_local_python_source("model")
)

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)

class ESC50Dataset(Dataset):
    def __init__(self,data_dir,metadata_file,split="train", transform = None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform
        
        if split =="train":
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]
            
        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(
            self.class_to_idx
        )
    
    def __len__(self):
        return len(self.metadata)
     
    def __getitem__(self,idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']
        
        waveform,sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0]>1: #[channel,samples] [2,44000] [1,44000]
            waveform = torch.mean(waveform,dim=0,keepdim=True)
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform
        return spectrogram, row['label']
                
def mixup_data(x,y): #x is features and y is labels
    lam = np.random.beta(0.4,0.4)  # Changed from 0.2,0.2 for more aggressive mixing
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # (0.7 * audio1) + (0.3 * audio2)
    mixed_x = lam * x +(1-lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
                
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)  # Improved mixup loss



                
@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume, "/models": model_volume},
    timeout=60 * 60 * 3
)
def train():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_directory = f' models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir = log_directory)
    
    
    
    
    
    
    esc50_dir = Path("/opt/esc50-data")
    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate = 22050,
            n_fft = 1024,
            hop_length = 512,
            n_mels = 128,
            f_min =0,
            f_max = 11025
    ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param = 40),  # Increased from 30
        T.TimeMasking(time_mask_param = 100),      # Increased from 80
        T.SpecAugment(freq_mask_param=40, time_mask_param=100, num_masks=2)  # Additional augmentation
        )
    
    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate = 22050,
            n_fft = 1024,
            hop_length = 512,
            n_mels = 128,
            f_min =0,
            f_max = 11025
    ),
        T.AmplitudeToDB()    
        )

    train_dataset = ESC50Dataset(
        data_dir = esc50_dir,
        metadata_file = esc50_dir / "meta" / "esc50.csv",
        split = 'train',
        transform = train_transform
    )
    
    val_dataset = ESC50Dataset(
        data_dir = esc50_dir,
        metadata_file = esc50_dir / "meta" / "esc50.csv",
        split = 'test',
        transform = val_transform)

    print("Training samples: ",len(train_dataset))
    print("Validation samples: ",len(val_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle = True, num_workers=2)  # Reduced batch size, added workers
    test_loader = DataLoader(val_dataset, batch_size=24, shuffle = False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes = len(train_dataset.classes), use_se=True, use_fpn=True)  # Enable new features
    model.to(device)
    
    num_epochs = 150  # Increased epochs
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # Increased label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.02)  # Adjusted hyperparameters
    
    # Improved learning rate scheduling
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=0.0015,  # Reduced max LR
        epochs=num_epochs, 
        steps_per_epoch=len(train_loader), 
        pct_start=0.15,  # Longer warmup
        anneal_strategy='cos'  # Cosine annealing
    )
    
    best_accuracy = 0.0
    patience = 15  # Early stopping patience
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss=0.0
        progress_bar = tqdm(train_loader, desc = f"Epoch{epoch+1}/{num_epochs}")
        for data, target in progress_bar:
            data,target = data.to(device),target.to(device) 
          
            if np.random.random() > 0.6:  # Increased mixup probability
                data, target_a, target_b, lam = mixup_data(data,target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                
            else:
                output = model(data)
                loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():4f}'})
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        
        #this is for validation after every epoch
        model.eval()
        
        correct = 0
        total =0
        val_loss = 0
        
        with torch.no_grad():
            for data,target in test_loader:
                data,target = data.to(device),target.to(device)
                outputs = model(data)
                loss = criterion(outputs,target)
                val_loss += loss.item()
                _,predicted = torch.max(outputs.data,1) #diff percentage scores for models
                total +=target.size(0)
                correct += (predicted == target).sum().item()
                
        accuracy =100 * correct / total
        avg_val_loss = val_loss/len(test_loader)
        
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        
        print(f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f} Validation Loss: {avg_val_loss:.4f} Accuracy: {accuracy:.2f}%')
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save({
                    'model_state_dict':model.state_dict(),
                    'accuracy':accuracy,
                    'epoch':epoch,
                    'classes': train_dataset.classes
                },
              "/models/best_models.pth")
            
            print(f'New best model saved with accuracy: {accuracy:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {patience} epochs without improvement')
                break
                
    writer.close()
    print(f"Training completed! Best accuracy: {best_accuracy:.2f}%")
    
           
            
            
            
            
            
                
            

@app.local_entrypoint()
def main():
    train.remote()
