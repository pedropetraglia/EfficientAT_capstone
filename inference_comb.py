import argparse
import torch
import librosa
import numpy as np
from torch import autocast
from contextlib import nullcontext

from models.MobileNetV3 import get_model as get_mobilenet, get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels
import matplotlib.pyplot as plt

import time
import os

#%%
def audio_tagging(args):
    """
    Running Inference on an audio clip.
    """
    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = args.audio_path
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    n_mels = args.n_mels
    width=args.model_width
    
    # load pre-trained model
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name,num_classes=4)
    model.to(device)
    model.eval()

   
    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()
    
    start_time = time.time()  

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform=length_adjust(waveform,sample_rate)
    waveform = torch.from_numpy(waveform[None, :]).to(device)
    
    # our models are trained in half precision mode (torch.float16)
    # run on cuda with torch.float16 to get the best performance
    # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
    
    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()
    #print(preds)
    
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    
    sorted_indexes = np.argsort(preds)[::-1]
   
    # Print audio tagging top probabilities
    print("\n************* Acoustic Event Detected: *****************")
    for k in range(4):
        print('{}: {:.3f}'.format(labels[sorted_indexes[k]],
            preds[sorted_indexes[k]]))
    print("********************************************************")


def length_adjust(signal,sr):
    if 0 < len(signal):                                          # workaround: 0 length causes error
              # ax1=plt.subplot(211)
              # ax1.plot(signal)
              signal, _ = librosa.effects.trim(signal,top_db=22) # trim, top_db=default(60)
             
     #make it unified length to 4 second
    if len(signal) > sr*4:     # long enough
                a =len(signal)/(sr*4)
                signal=librosa.effects.time_stretch(signal, rate=a)
             
    elif len(signal) < sr*4:       # repeat itself until 4 second
                repts = int(sr*4/len(signal))
                rem=sr*4%len(signal)
                signal=np.append(np.tile(signal,repts),signal[0:rem])
    else:
        pass
    
    # ax2=plt.subplot(212)
    # ax2.plot(signal)       
    return signal

#%%
if __name__ == '__main__':
    
    fname="pa (1).wav"
    mname="mn40_multi2"
    
    
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model name decides, which pre-trained model is loaded
    parser.add_argument('--model_name', type=str, default=f'{mname}')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--audio_path', type=str, default=f"resources/{fname}")

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=512)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--model_width', type=float, default=3.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    
    # overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
    parser.add_argument('--ensemble', nargs='+', default=[])

    args = parser.parse_args()
    single_test=0

    if single_test==1:
        audio_tagging(args)
    else:
    
#%%
    
        li=os.listdir('resources/')
        li[:]=[name for name in li if name[-4:]=='.wav']
        li.remove("metro_station-paris.wav")
            
        sample_rate=22050
    
        waves=np.empty((0, sample_rate*4), int)
        for name in li:
    
            (waveform, _) = librosa.core.load('resources/'+name, sr=sample_rate, mono=True)
            waveform=length_adjust(waveform,sample_rate)
            waves=np.append(waves, waveform[None,:],axis=0)
    
        waves = (torch.from_numpy(waves)).type(torch.FloatTensor)
       
        model = get_mobilenet(width_mult=NAME_TO_WIDTH(mname), pretrained_name=mname,num_classes=4) 
       
        # our models are trained in half precision mode (torch.float16)
        # run on cuda with torch.float16 to get the best performance
        # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
        start_time = time.time()  
        
        mel = AugmentMelSTFT(n_mels=128, sr=sample_rate, win_length=1024, hopsize=512)
        mel.eval()
        
        with torch.no_grad():
            spec = mel(waves)
            preds, features = model(spec.unsqueeze(1))
        preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()
        print('\n',time.time()  -start_time)
         
        scores=np.round(preds,3)
        scores_table=dict(zip(li,scores))
         
        del (spec,preds,waveform,waves,name,start_time)
        
        
        







        
    
