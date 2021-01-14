import torch
import torchaudio
from torch.optim import lr_scheduler
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

# RuntimeError: CUDA error: unspecified launch failure
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def main(learning_rate=5e-4,
         batch_size=5,
         epochs=1,
         experiment=Experiment(api_key='dummy_key', disabled=True)):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 7,
        "rnn_dim": 1024,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
    experiment.log_parameters(hparams)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = Aduio_DataLoader(
        data_folder=r'D:\dataset\ntut-ml-2020-taiwanese-e2e\train',
        utterance_csv=
        r'D:\dataset\ntut-ml-2020-taiwanese-e2e\train-toneless_update.csv',
        sr=16000,
        dimension=480000)
    val_dataset = Aduio_DataLoader(
        data_folder=r'D:\dataset\ntut-ml-2020-taiwanese-e2e\val',
        utterance_csv=
        r'D:\dataset\ntut-ml-2020-taiwanese-e2e\train-toneless_update.csv',
        sr=16000,
        dimension=480000)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        collate_fn=lambda x: data_processing(x, 'train'),
        num_workers=0,
        pin_memory=True)

    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=hparams['batch_size'],
        shuffle=False,
        collate_fn=lambda x: data_processing(x, 'val'),
        num_workers=0,
        pin_memory=True)
    model = SpeechRecognitionModel(hparams['n_cnn_layers'],
                                   hparams['n_rnn_layers'], hparams['rnn_dim'],
                                   hparams['n_class'], hparams['n_feats'],
                                   hparams['stride'],
                                   hparams['dropout']).to(device)
    # print(model)
    print('Num Model Parameters',
          sum([param.nelement() for param in model.parameters()]))
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
    #                                            T_max=hparams['epochs'],
    #                                            eta_min=1e-6,
    #                                            last_epoch=-1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=hparams['learning_rate'],
                                              steps_per_epoch=int(
                                                  len(train_loader)),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='cos')

    scaler = torch.cuda.amp.GradScaler()
    train_data_len = len(train_loader.dataset)
    train_epoch_size = math.ceil(train_data_len / batch_size)
    val_data_len = len(val_loader.dataset)
    val_epoch_size = math.ceil(val_data_len / batch_size)
    iter_meter = IterMeter()
    train_losses, val_losses, wers, lrs = [], [], [], []
    for epoch in range(1, epochs + 1):
        print('running epoch: {} / {}'.format(epoch, epochs))
        start_time = time.time()
        # 訓練模式
        model.train()
        train_loss = 0
        with tqdm(total=train_epoch_size,
                  desc='train',
                  postfix=dict,
                  mininterval=0.3) as pbar:
            with experiment.train():
                for batch_idx, _data in enumerate(train_loader):
                    spectrograms, labels, input_lengths, label_lengths, _ = _data
                    # print(input_lengths)
                    spectrograms, labels = spectrograms.to(device), labels.to(
                        device)
                    optimizer.zero_grad()
                    output = model(spectrograms)
                    output = F.log_softmax(output, dim=2)
                    output = output.transpose(0, 1)
                    loss = criterion(output, labels, input_lengths,
                                     label_lengths)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    experiment.log_metric('loss',
                                          loss.item(),
                                          step=iter_meter.get())
                    experiment.log_metric('learning_rate',
                                          scheduler.get_last_lr(),
                                          step=iter_meter.get())
                    iter_meter.step()
                    waste_time = time.time() - start_time
                    train_loss += loss.item() * spectrograms.size(0)
                    pbar.set_postfix(
                        **{
                            'loss': loss.item(),
                            'lr': round(scheduler.get_last_lr()[0], 6),
                            'step/s': waste_time
                        })
                    pbar.update(1)
                    start_time = time.time()
                    lrs.append(scheduler.get_last_lr()[0])

        start_time = time.time()
        # 評估模式
        model.eval()
        val_loss = 0
        val_cer, val_wer = [], []
        with tqdm(total=val_epoch_size,
                  desc='val',
                  postfix=dict,
                  mininterval=0.3) as pbar:
            with experiment.test():
                with torch.no_grad():
                    for I, _data in enumerate(val_loader):
                        spectrograms, labels, input_lengths, label_lengths, _ = _data
                        spectrograms, labels = spectrograms.to(
                            device), labels.to(device)
                        output = model(spectrograms)  # (batch, time, n_class)
                        output = F.log_softmax(output, dim=2)
                        output = output.transpose(0,
                                                  1)  # (time, batch, n_class)
                        loss = criterion(output, labels, input_lengths,
                                         label_lengths)
                        val_loss += loss.item() * spectrograms.size(0)
                        decoded_preds, decoded_targets = GreedyDecoder(
                            output.transpose(0, 1), labels, label_lengths)
                        for j in range(len(decoded_preds)):
                            # val_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                            val_wer.append(
                                wer(reference=decoded_targets[j],
                                    hypothesis=decoded_preds[j]))
                        waste_time = time.time() - start_time
                        pbar.set_postfix(**{
                            'loss': loss.item(),
                            'step/s': waste_time
                        })
                        pbar.update(1)
                        start_time = time.time()
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # avg_cer = sum(val_cer) / len(val_cer)
        avg_wer = sum(val_wer) / len(val_wer)
        wers.append(avg_wer)

        experiment.log_metric('val_loss', val_loss, step=iter_meter.get())
        # experiment.log_metric('cer', avg_cer, step=iter_meter.get())
        experiment.log_metric('wer', avg_wer, step=iter_meter.get())
        # print(
        #     'Val set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'
        #     .format(val_loss, avg_cer, avg_wer))
        print(
            'average train_loss: {:.4f}, average val_loss: {:.4f}, average wer: {:.4f}\n'
            .format(train_loss, val_loss, avg_wer))
        torch.save(
            model.state_dict(),
            './logs/epoch%d-train_loss%.4f-val_loss%.4f-avg_wer%.4f.pth' %
            (epoch, train_loss, val_loss, avg_wer))

    # 繪製圖
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(loc='best')
    plt.savefig('./images/loss.jpg')
    plt.show()

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('WER')
    plt.plot(wers, label='WER')
    plt.legend(loc='best')
    plt.savefig('./images/wer.jpg')
    plt.show()

    plt.figure()
    plt.xlabel('Mini-batch')
    plt.ylabel('Learning Rate')
    plt.plot(lrs, label='Learning Rate')
    plt.legend(loc='best')
    plt.savefig('./images/lr.jpg')
    plt.show()


if __name__ == "__main__":
    main()
