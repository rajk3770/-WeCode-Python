from gmm import read_wav, save_model, get_pos_feat
from scipy.io import wavfile
import numpy as np
import random
import os
import pickle
from vad import write_vad

# simulate dialogue of speaker A and B
def generate_mix():

    data = []
    segs = []

    fs, sig = read_wav('A.wav')
    fs, sig1 = read_wav('B.wav')

    for i in range(60):
        is_true = random.choice([True, False])
        dur = random.uniform(0.5, 4)

        if is_true:
            s = random.randint(0, sig.shape[0]-2*fs)
            data.extend(sig[s:int(s+dur*fs), ])

        else:
            s = random.randint(0, sig1.shape[0]-2*fs)
            data.extend(sig1[s:int(s+dur*fs), ])

        if len(segs) == 0:
            segs.append([0, dur, 'A' if is_true else 'B'])
        else:
            segs.append([segs[-1][1], segs[-1][1]+dur,
                         'A' if is_true else 'B'])

    wavfile.write("mix.wav", fs, np.asarray(data))
    return segs

def vadSeg(gmm_path,wav):
    write_vad(wav,'tmp.wav')
    segment(gmm_path,'tmp.wav')
    os.remove('tmp.wav')

def segment(gmm_path, wav):

    with open(gmm_path, 'rb') as f:
        gmm = pickle.load(f)

    fs, sig = read_wav(wav)

    scores = []
    segs = []
    i = 0
    while True:

        s = int(i*0.5*fs)
        e = int((i*0.5+2)*fs)
        if e > sig.shape[0]:
            break
        seg = sig[s:e, ]

        feat = get_pos_feat(seg, fs)
        scores.append(gmm.score(feat))

        if scores[-1] < -49:
            segs.append([i*0.5, i*0.5+2])
        i += 1
    final_segs = []
    final_segs.append(segs[0])

    for i in range(1, len(segs)):
        if round(segs[i][1]-0.5, 1) == round(final_segs[-1][1], 1):
            final_segs[-1][1] += 0.5
        else:
            final_segs.append(segs[i])

    for i in range(len(final_segs)):
        s = final_segs[i][0]
        e = final_segs[i][1]
        final_segs[i][0] = round(s+1, 2)
        final_segs[i][1] = round(e-1, 2)

    out_wav = []

    for f in final_segs:
        print(fs)
        print(f[0],f[1])
        out_wav.extend(sig[int(f[0])*fs:int(f[1])*fs])
    out_wav = np.asarray(out_wav)

    wavfile.write('out.wav',100,out_wav)

    return final_segs

# evaluate the purity of extracted voices


def eval():
    segs = generate_mix()
    cut_segs = segment('train_A.mdl', 'mix.wav')
    cut_real_user = 0
    real_user = 0
    for s, e in cut_segs:
        for s1, e1, id in segs:
            if id != 'A':
                if s <= s1 and e >= e1:
                    cut_real_user += (e1-s1)
                elif s > s1 and e < e1:
                    cut_real_user += (e-s)
                elif s < s1 and e >= s1 and e <= e1:
                    cut_real_user += (e-s1)
                elif s >= s1 and s <= e1 and e > e1:
                    cut_real_user += (e1-s)
    for s, e, id in segs:
        if id != 'A':
            real_user += (e-s)
    print('real B:', real_user)
    print('extract B:', cut_real_user)
    print('extract A:', float(os.popen('soxi -D %s' %
                                       'out.wav').read())-cut_real_user)
