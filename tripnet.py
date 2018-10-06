import os
import scipy.io.wavfile as wavfile
import numpy as np
import librosa
import random
from vad import write_vad
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def choose_spks():
    path = 'data_thchs30/data'
    train_path = 'data_thchs30/train'
    test_path = 'data_thchs30/test'

    spks = ['A8','B15','C7']

    for spk in spks:
        wavs = os.popen('ls %s/%s*'%(path,spk)).read().split()
        os.system('mkdir -p %s/%s'%(train_path,spk))
        os.system('mkdir -p %s/%s'%(test_path,spk))
        train = wavs[:-10]
        test = wavs[-10:]
        for t in train:
            write_vad(t,t.replace('data/','train/'+spk+'/'))
        for t in test:
            write_vad(t,t.replace('data/','test/'+spk+'/'))

def feat_extractor():
    feats = []
    for spk in ['A','B','C']:
        sig = librosa.load('data_thchs30/train/%s.wav'%spk)[0]
        sample_rate = 16000
        duration = 0.032
        step = 0.02
        mfcc = librosa.feature.mfcc(y=sig,sr=sample_rate,n_mfcc=12,
            n_fft=int(duration*sample_rate),hop_length=int(step*sample_rate),
            n_mels=40,htk=True,fmin=0)
        mfcc_d = librosa.feature.delta(mfcc, width=9, order=1, axis=-1)
        mfcc_dd = librosa.feature.delta(mfcc, width=9, order=2, axis=-1)
        feats.append(np.vstack([mfcc,mfcc_d,mfcc_dd]))
    np.savez('data_thchs30/spk_train.npz',A=feats[0],B=feats[1],C=feats[2])
    
class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.rnn =  nn.LSTM(3600,64,1)
        self.l1 =   nn.Linear(64, 16)
        self.tan1 = nn.Tanh()
        self.l2 =   nn.Linear(16, 16)
        self.tan2 = nn.Tanh()


    def init_hidden(self):
        self.h0 = torch.zeros(1,1,64)
        self.c0 = torch.zeros(1,1,64)

    def forward(self,x):
        out,(self.h0,self.c0) = self.rnn(x,(self.h0,self.c0))
        
        out = F.normalize(out)

        out = self.tan1(self.l1(out))

        out = self.tan2(self.l2(out))

        return F.normalize(out)

def getTrip(mode='train'):
    
    if mode == 'train':
        feats = np.load('data_thchs30/spk_train.npz')
    else:
    	feats = np.load('data_thchs30/spk_test.npz')

    spks = ['A','B','C']

    for spk in spks:
       
        while True:
          neg = random.choice(spks)
          if neg != spk:
             break 
   
        spk_len = feats[spk].shape[1]
        neg_len = feats[neg].shape[1]

        start = 0

        while True:
            
            if start+100 > spk_len:
            	break

            pos_start = random.randint(0,spk_len-100)
            neg_start = random.randint(0,neg_len-100)
          
            a = feats[spk].transpose()[start:start+100,].reshape(1,3600)
            p = feats[spk].transpose()[pos_start:pos_start+100,].reshape(1,3600)
            n = feats[neg].transpose()[neg_start:neg_start+100,].reshape(1,3600)

            start += 50

            yield (torch.from_numpy(a).resize_(1,1,3600).float(),
                torch.from_numpy(p).resize_(1,1,3600).float(),
                torch.from_numpy(n).resize_(1,1,3600).float())

def train():
    net = Net()
   # net.load_state_dict(torch.load('net.pkl'))
    
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    trip_loss = nn.TripletMarginLoss(margin=0.2)

    for epoch in range(1,11):
        for (anc,pos,neg) in getTrip('train'):
            net.zero_grad()
            net.init_hidden()
            x = net(anc)
            y = net(pos)
            z = net(neg)
            #print(x,y,z)
            loss = trip_loss(x,y,z)
            loss.backward(retain_graph=True)
            opt.step()
            print(loss.data)
        torch.save(net.state_dict(), 'net.pkl')

def test():
    net = Net()
    trip_loss = nn.TripletMarginLoss(margin=0.2)
    
    for i in range(10):
        net.load_state_dict(torch.load('net.pkl'))
        net.init_hidden()
        for (anc,pos,neg) in getTrip('test'):
            loss = trip_loss(net(anc),net(pos),net(neg))
            print(loss.data)
train()