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


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, train_loader, test_loader, criterion, optimizer,
          scheduler, epochs, epoch, train_epoch_size, val_epoch_size,
          iter_meter, experiment):
    print('running epoch: {} / {}'.format(epoch, epochs))
    start_time = time.time()
    # 訓練模式
    model.train()
    data_len = len(train_loader.dataset)
    with tqdm(total=train_epoch_size,
              desc='train',
              postfix=dict,
              mininterval=0.3) as pbar:
        with experiment.train():
            for batch_idx, _data in enumerate(train_loader):
                spectrograms, labels, input_lengths, label_lengths = _data
                spectrograms, labels = spectrograms.to(device), labels.to(
                    device)
                optimizer.zero_grad()
                output = model(spectrograms)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # (time, batch, n_class)
                loss = criterion(output, labels, input_lengths, label_lengths)

                loss.backward()
                experiment.log_metric('loss',
                                      loss.item(),
                                      step=iter_meter.get())
                experiment.log_metric('learning_rate',
                                      scheduler.get_last_lr(),
                                      step=iter_meter.get())
                optimizer.step()
                scheduler.step()
                iter_meter.step()

                waste_time = time.time() - start_time
                pbar.set_postfix(
                    **{
                        'total_loss': loss.item(),
                        'lr': round(scheduler.get_last_lr()[0], 5),
                        'step/s': waste_time
                    })
                pbar.update(1)
                start_time = time.time()

    start_time = time.time()
    # 評估模式
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with tqdm(total=val_epoch_size, desc='val', postfix=dict,
              mininterval=0.3) as pbar:
        with experiment.test():
            with torch.no_grad():
                for I, _data in enumerate(test_loader):
                    spectrograms, labels, input_lengths, label_lengths = _data
                    spectrograms, labels = spectrograms.to(device), labels.to(
                        device)
                    output = model(spectrograms)  # (batch, time, n_class)
                    output = F.log_softmax(output, dim=2)
                    output = output.transpose(0, 1)  # (time, batch, n_class)
                    loss = criterion(output, labels, input_lengths,
                                     label_lengths)
                    test_loss += loss.item() / len(test_loader)
                    decoded_preds, decoded_targets = GreedyDecoder(
                        output.transpose(0, 1), labels, label_lengths)
                    for j in range(len(decoded_preds)):
                        # test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                        test_wer.append(
                            wer(reference=decoded_targets[j],
                                hypothesis=decoded_preds[j]))
                    waste_time = time.time() - start_time
                    pbar.set_postfix(**{
                        'total_loss': loss.item(),
                        'step/s': waste_time
                    })
                    pbar.update(1)
                    start_time = time.time()
    # avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    experiment.log_metric('test_loss', test_loss, step=iter_meter.get())
    # experiment.log_metric('cer', avg_cer, step=iter_meter.get())
    experiment.log_metric('wer', avg_wer, step=iter_meter.get())
    # print(
    #     'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'
    #     .format(test_loss, avg_cer, avg_wer))
    print('Test set: Average loss: {:.4f}, Average WER: {:.4f}\n'.format(
        test_loss, avg_wer))
    torch.save(
        model.state_dict(), './logs/epoch%d-val_loss%.4f-avg_wer%.4f.pth' %
        (epoch, total_loss / test_loss, avg_wer))


def main(learning_rate=5e-4,
         batch_size=10,
         epochs=10,
         train_url="train-clean-100",
         test_url="test-clean",
         experiment=Experiment(api_key='dummy_key', disabled=True)):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
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
    if not os.path.isdir("./data"):
        os.makedirs("./data")
    train_dataset = torchaudio.datasets.LIBRISPEECH("./data",
                                                    url=train_url,
                                                    download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("./data",
                                                   url=test_url,
                                                   download=True)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        collate_fn=lambda x: data_processing(x, 'train'),
        num_workers=0,
        pin_memory=True)
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=hparams['batch_size'],
        shuffle=False,
        collate_fn=lambda x: data_processing(x, 'valid'),
        num_workers=0,
        pin_memory=True)
    model = SpeechRecognitionModel(hparams['n_cnn_layers'],
                                   hparams['n_rnn_layers'], hparams['rnn_dim'],
                                   hparams['n_class'], hparams['n_feats'],
                                   hparams['stride'],
                                   hparams['dropout']).to(device)
    print(model)
    print('Num Model Parameters',
          sum([param.nelement() for param in model.parameters()]))
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=hparams['learning_rate'],
                                              steps_per_epoch=int(
                                                  len(train_loader)),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')
    train_data_len = len(train_loader.dataset)
    train_epoch_size = math.ceil(train_data_len / batch_size)
    val_data_len = len(test_loader.dataset)
    val_epoch_size = math.ceil(val_data_len / batch_size)
    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, test_loader, criterion, optimizer,
              scheduler, epochs, epoch, train_epoch_size, val_epoch_size,
              iter_meter, experiment)


if __name__ == "__main__":
    main()