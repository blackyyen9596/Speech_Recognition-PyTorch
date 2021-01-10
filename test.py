import torchaudio
import librosa

sample_rate = 1000
filepath = r'D:\dataset\ntut-ml-2020-spring-taiwanese-e2e\test-shuf\1.wav'
wav, sr = librosa.load(filepath, sr=sample_rate)
test_audio_transforms = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate, n_mels=128)
# mfccs = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_mels=128)

# spec = test_audio_transforms(wav).squeeze(0).transpose(0, 1)