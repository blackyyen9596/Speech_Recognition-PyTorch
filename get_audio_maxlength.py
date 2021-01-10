import librosa
import os
from tqdm import tqdm

path = r'D:\dataset\ntut-ml-2020-spring-taiwanese-e2e\train_org'
filelist = os.listdir(path)
maxduration = 0
for filename in (filelist):
    file_path = os.path.join(path, filename)
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y, sr)
    if duration > maxduration:
        maxduration = duration
        print(maxduration)
        print(filename)

