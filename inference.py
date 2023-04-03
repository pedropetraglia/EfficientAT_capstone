from sklearn import metrics
import argparse
import torch
import librosa
import numpy as np
from torch import autocast
from contextlib import nullcontext
import os
from pandas import *
import csv
import matplotlib.pyplot as plt

from models.MobileNetV3 import get_model as get_mobilenet, get_ensemble_model
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels


def audio_tagging(args, pathh):
    """
    Running Inference on an audio clip.
    """
    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    audio_path = pathh
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    n_mels = args.n_mels

    # load pre-trained model
    if len(args.ensemble) > 0:
        model = get_ensemble_model(args.ensemble)
    else:
        model = get_mobilenet(width_mult=NAME_TO_WIDTH(model_name), pretrained_name=model_name, strides=args.strides,
                              head_type=args.head_type)
    model.to(device)
    model.eval()

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=n_mels, sr=sample_rate, win_length=window_size, hopsize=hop_size)
    mel.to(device)
    mel.eval()

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = torch.from_numpy(waveform[None, :]).to(device)

    # our models are trained in half precision mode (torch.float16)
    # run on cuda with torch.float16 to get the best performance
    # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse
    with torch.no_grad(), autocast(device_type=device.type) if args.cuda else nullcontext():
        spec = mel(waveform)
        preds, features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    sorted_indexes = np.argsort(preds)[::-1]

    # Print audio tagging top probabilities
    print("************* Acoustic Event Detected: *****************")
    for k in range(4):
        print('{}: {}'.format(labels[k],
            preds[k]))
    print("********************************************************")

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    # model name decides, which pre-trained model is loaded
    parser.add_argument('--model_name', type=str, default='mn40-98')
    parser.add_argument('--strides', nargs=4, default=[2, 2, 2, 2], type=int)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--audio_path', type=str, default='resources/siren-03-26.wav', required=False)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_mels', type=int, default=128)

    # overwrite 'model_name' by 'ensemble_model' to evaluate an ensemble
    parser.add_argument('--ensemble', nargs='+', default=[])

    args = parser.parse_args()


    ######## THRESHOLD CALCULATION ###########################

    predicted = [0.29, 0.86, 0.11, 0.28] #test data

    targets = []

    path = r"C:/Users/Pedro/Downloads/combined_dataset_and_csv/new2/data/"
    dir_list = sorted(os.listdir(path))
    num_dir = len(dir_list)

    data = read_csv(r"C:/Users/Pedro/Downloads/combined_dataset_and_csv/new2/combined_finetuning.csv")
    outputs = np.append(data['car_horn'].tolist(), data['car_passing'].tolist())
    outputs = np.append(outputs, data['engine'].tolist())
    outputs = np.append(outputs, data['siren'].tolist())

    holding = []
    holding2 = []
    outputs = []
    hold = []
    accuracy = []

    with open(r"C:/Users/Pedro/Downloads/combined_dataset_and_csv/new2/combined_finetuning.csv") as file_obj:
        heading = next(file_obj)
        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            for i in range(0,4):
                holding = np.append(holding, int(row[i]))
            outputs.append(holding.tolist())
            holding = []
    print(outputs)

    for probability_threshold in np.arange(1.00, 0.94, -0.02):
        for file in range(0, num_dir):
            actual_predicts = audio_tagging(args, pathh=path+dir_list[file])
            for i in range(0, 4):
                if actual_predicts[i] > probability_threshold:
                    holding2 = np.append(holding2, int(1))
                else:
                    holding2 = np.append(holding2, int(0))
            targets.append(holding2.tolist())
            holding2 = []
            print(targets)



        temp_acc = metrics.accuracy_score(targets, outputs)

        accuracy = np.append(accuracy, temp_acc)
        hold.append(probability_threshold.tolist())

        targets = []

    print(accuracy)
    print(hold)

    plt.plot(hold, accuracy, label=f'HOLD BY ACCURACY')
    #plt.plot(iterations, mn40_predict_time, label=f'mn40 prediction time, avg: {round(mn40_avg, 2)} s')

    plt.xlabel('Prediction Threshold for <MODEL>')
    plt.ylabel('Accuracy')
    #plt.xticks(range(20))
    plt.title('Accuracy vs Prediction Threshold for <MODEL>')
    plt.legend(loc='best')
    plt.savefig('prediction_threshold.png')
    plt.show()





#print("targets: "+str(targets))
#print("outputs: "+str(outputs))
#print("accuracy: "+str(accuracy))

#y_pred = [0, 2, 1, 3]

#y_pred = [[0], [0, 1]]
#y_true = [[0, 1], [0, 2]]



    ############ THRESHOLD CALCULATION ######################