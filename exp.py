import matplotlib.pyplot as plt
import numpy as np
from seg import generate_mix, segment
from  python_speech_features.base import mfcc,delta
from scipy.io import wavfile

def plot():

 
    A = generate_mix()
    B = segment('train_A.mdl', 'mix.wav')
    plt.figure(figsize=(10, 3))

    for i in range(len(B)):
        idx = len(B)-i-1
        plt.barh(2, B[idx][1], color='g')
        plt.barh(2, B[idx][0], color='w')

    for i in range(len(A)):
        idx = len(A)-i-1
        plt.barh(1, A[idx][1], color='r' if A[idx][2] == 'A' else 'g')
    
    plt.legend(("user","service"))

    ax = plt.gca()
    plt.setp( ax.get_yticklabels(), visible=False)
    ax.yaxis.set_ticks_position('none') 
    leg = ax.get_legend()
    leg.legendHandles[1].set_color('red')
    
    plt.savefig("demo.jpg") 

if __name__ == '__main__':
    plot()
