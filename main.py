
import modal 
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram,AmplitudeToDB
import torchaudio.transforms as T
from model import AudioCNN
from pydantic import BaseModel
import soundfile as sf
import base64
import io
import numpy as np


app = modal.App("audio-cnn-inference")
image =(modal.Image.debian_slim()
        .pip_install_from_requirements("requirements.txt")
        .apt_install(["libsndfile1"])
        .add_local_python_source('model'))

model_volume = modal.Volume.from_name("esc-model")

class AudioProcessor:
  def __init__(self):
    self.transform=nn.Sequential(
        MelSpectrogram(
            sample_rate = 22050,
            n_fft = 1024,
            hop_length = 512,
            n_mels = 128,
            f_min =0,
            f_max = 11025
    ),
        T.AmplitudeToDB(),
        
    )

def process_audio_chunk(self,audio_data):
  waveform = torch.from_numpy(audio_data).float()
  waveform = waveform.unsqueeze(0)
  spectogram = self.transform(waveform)
  return spectogram.unsqueeze(0)

class InferenceRequest():
  audio_data: str

@app.cls(image=image,gpu="A10G",volumes={"/models":model_volume},scaledown_window = 0)
class AudioClassifier:
  @modal.enter()
  def load_model(self):
    print("Loading model...")
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("/models/best_models.pth", map_location=self.device)
    self.classes = checkpoint['classes']
    self.model = AudioCNN(num_classes = len(self.classes))
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.model.to(self.device)
    self.model.eval()
    self.audio_processor = AudioProcessor()
    print("Model loaded successfully on enter")
    
    
  @modal.fastapi_function(gpu="A10G")
  def inference(self,request:InferenceRequest):
    # prod - frontend -> upload file to s3 -> inference endpoint -> download frm s3 bucket 
    audio_bytes = base64.b64decode(request.audio_data)
    
    audio_data,sample_rate=sf.read(io.BytesIO(audio_bytes),dtype='float32')
    if audio_data.ndim>1:
      audio_data = np.mean(audio_data,axis=1)
    




