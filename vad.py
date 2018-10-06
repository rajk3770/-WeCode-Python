import sys,os,random
import scipy.io.wavfile as wavfile
import numpy as np
import time

def remove_silence(fs, signal,
        frame_duration = 0.02,
        frame_shift = 0.01,
        perc = 0.15):

    orig_dtype = type(signal[0])
    typeinfo = np.iinfo(orig_dtype)
    is_unsigned = typeinfo.min >= 0
    signal = signal.astype(np.int64)

    if is_unsigned:
        signal = signal - (typeinfo.max + 1) / 2

    siglen = len(signal)
    retsig = np.zeros(siglen, dtype = np.int64)
    frame_length = int(frame_duration * fs)
    frame_shift_length = int(frame_shift * fs)
    new_siglen = 0
    i = 0
  
    average_energy = np.sum(signal ** 2) / float(siglen)
 
    while i < siglen:
        subsig = signal[i:i + frame_length]
        ave_energy = np.sum(subsig ** 2) / float(len(subsig))
        if ave_energy < average_energy * perc:
            i += frame_length
        else:
            sigaddlen = min(frame_shift_length, len(subsig))
            retsig[new_siglen:new_siglen + sigaddlen] = subsig[:sigaddlen]
            new_siglen += sigaddlen
            i += frame_shift_length
    retsig = retsig[:new_siglen]
    
    if is_unsigned:
        retsig = retsig + typeinfo.max / 2
    return retsig.astype(orig_dtype)

def write_vad(fpath, new_fpath):
    fs, signal = wavfile.read(fpath)
    signal_out = remove_silence(fs, signal)
    wavfile.write(new_fpath, fs, signal_out)
    return fpath

def get_vad(fpath):
    fs, signal = wavfile.read(fpath)
    signal_out = remove_silence(fs, signal)
    duration = signal_out.size / fs
    return signal_out, duration


