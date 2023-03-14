import torch
import pandas as pd
import librosa
import numpy as np
import time

from .models.MobileNetV3 import get_model as get_mobilenet
from .models.preprocess import AugmentMelSTFT
from .helpers.utils import NAME_TO_WIDTH, labels

class mn40_eval():
    def __init__(self):
        self.device = torch.device('cpu')
        
        self.model = get_mobilenet(width_mult=NAME_TO_WIDTH('mn30_urban'), pretrained_name='mn30_urban',num_classes=3)
        self.model.to(self.device)
        self.model.eval()
        
        # model to preprocess waveform into mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=128, sr=16000, win_length=800, hopsize=320)
        self.mel.to(self.device)
        self.mel.eval()
        
    def predict(self, path):
        start_time = time.time()
        (waveform, _) = librosa.core.load(path, sr=16000, mono=True)
        waveform = torch.from_numpy(waveform[None, :]).to(self.device)

        # our models are trained in half precision mode (torch.float16)
        # run on cuda with torch.float16 to get the best performance
        # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
        
        spec = self.mel(waveform)
        preds, features = self.model(spec.unsqueeze(0))
        preds = torch.sigmoid(preds.float()).squeeze().cpu().detach().numpy()
        
        end_time = time.time()
        
        
        sorted_indexes = np.argsort(preds)[::-1]
    
        # Print audio tagging top probabilities
        print("************* Acoustic Event Detected: *****************")
        for k in range(3):
            print('{}: {:.3f}'.format(labels[sorted_indexes[k]],
                preds[sorted_indexes[k]]))
        print("********************************************************")
        
        return end_time - start_time 
        
if __name__ == '__main__':
    model = mn40_eval()
    path = ""
    model.predict(path)