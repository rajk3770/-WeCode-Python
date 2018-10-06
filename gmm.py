import pickle,os
import numpy as np

from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture
from vad import write_vad

def read_wav(wav):
    return wavfile.read(wav)


def get_pos_feat(sig, fs):
    return mfcc(sig, fs)


def get_feat(wav):
    fs, sig = read_wav(wav)
    mfcc_feature = mfcc(sig, fs)
    return mfcc_feature


def fit(src, mixture=32):
    gmm = GaussianMixture(mixture)
    if type(src) == str:
        src = get_feat(src)
    gmm.fit(src)
    return gmm


def score(gmm, feat):
    return gmm.score(feat)


def save_model(path,wav):
    write_vad(wav,'tmp.wav')
    wav = 'tmp.wav'
    gmm = fit(wav)
    with open(path, 'wb') as f:
        pickle.dump(gmm, f)
    os.remove('tmp.wav')


def predict(path, wav):
    write_vad(wav,'tmp.wav')
    wav = 'tmp.wav'
    with open(path, 'rb') as f:
        gmm = pickle.load(f)
    feat = get_feat(wav)
    os.remove('tmp.wav')
    return score(gmm, feat)


