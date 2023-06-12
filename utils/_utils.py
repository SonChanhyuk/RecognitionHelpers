import librosa
from pydub import AudioSegment, effects
import numpy as np

emotion_dic = {
    'neutral' : 0,
    'happy'   : 1,
    'sad'     : 2, 
    'angry'   : 3, 
    'fear'    : 4, 
    'disgust' : 5
}

def encode(label):
    return emotion_dic.get(label)

def decode(label):
    return list(emotion_dic.keys())[label]

def preprocess_audio(path):
    _, sr = librosa.load(path)
    raw_audio = AudioSegment.from_file(path)
    
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    if (len(trimmed)>180000):
        padded = trimmed[:180000]
    else:
        padded = np.pad(trimmed, (0, 180000-len(trimmed)), 'constant')

    FRAME_LENGTH = 2048
    HOP_LENGTH = 512

    try: 
        zcr = librosa.feature.zero_crossing_rate(padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=padded, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        mfcc = librosa.feature.mfcc(y=padded, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    except:
        print(f"Failed for path: {path}")

    X = np.concatenate((
        np.swapaxes(zcr, 0, 1), 
        np.swapaxes(rms, 0, 1), 
        np.swapaxes(mfcc, 0, 1)), 
        axis=1
    )
    return X.astype('float32')

