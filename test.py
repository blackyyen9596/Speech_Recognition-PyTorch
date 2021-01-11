import torch
import torchaudio
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from nets.model import SpeechRecognitionModel
from comet_ml import Experiment
from utils.processing import TextTransform, data_processing, GreedyDecoder
import os
import torch.nn.functional as F
from utils.wer import levenshtein_distance as wer
from tqdm import tqdm
import time
import math
from audio_dataloader import Aduio_DataLoader
import matplotlib.pyplot as plt
import pandas as pd

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 29,
    "n_feats": 128,
    "stride": 2,
    "dropout": 0.1,
    "batch_size": 1,
}

model = SpeechRecognitionModel(hparams['n_cnn_layers'],
                               hparams['n_rnn_layers'], hparams['rnn_dim'],
                               hparams['n_class'], hparams['n_feats'],
                               hparams['stride'],
                               hparams['dropout']).to(device)
model.load_state_dict(
    torch.load(r'./weights/epoch75-val_loss0.2958-avg_wer0.0519.pth'))
test_dataset = Aduio_DataLoader(
    r'D:\dataset\ntut-ml-2020-spring-taiwanese-e2e\test-shuf')

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=hparams['batch_size'],
                              shuffle=False,
                              collate_fn=lambda x: data_processing(x, 'test'),
                              num_workers=0,
                              pin_memory=True)
test_data_len = len(test_loader.dataset)
test_epoch_size = math.ceil(test_data_len / hparams['batch_size'])
start_time = time.time()
file_list, pred_list = [], []
with tqdm(total=test_epoch_size, desc='test', postfix=dict,
          mininterval=0.3) as pbar:
    with torch.no_grad():
        for I, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths, filename = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)
            decoded_preds, decoded_targets = GreedyDecoder(
                output.transpose(0, 1),
                labels.cpu().numpy().astype(int), label_lengths)
            pred = decoded_preds[0].replace("'", " ").strip()  #刪除前後的空格
            file_list.append(filename)
            pred_list.append(pred)
            waste_time = time.time() - start_time
            pbar.set_postfix(**{'step/s': waste_time})
            pbar.update(1)

id_path = r'D:\dataset\ntut-ml-2020-spring-taiwanese-e2e\sample.csv'
save_path = r'D:\dataset\ntut-ml-2020-spring-taiwanese-e2e\test.csv'
dictionary = {'id': file_list[:], 'text': pred_list[:]}
final_file_list, final_pred_list = [], []
for i in range(len(file_list)):
    final_file_list.append(i + 1)
    location = file_list.index(str(i + 1))
    final_pred_list.append(str(dictionary.get('text')[location]))
my_submission = pd.DataFrame({
    'id': final_file_list[:],
    'text': final_pred_list[:]
})
my_submission.to_csv(save_path, index=False)