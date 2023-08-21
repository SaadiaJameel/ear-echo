import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,fftfreq,ifft
from scipy import signal
from scipy.special import rel_entr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import librosa
import librosa.display

F_min = 17000 # 16KHz
F_max = 23000 # 22KHz
sampling_rate = 100000 # 100KHz
Fs = 100000  # 100KHz
FMCW_duration = 0.08
silent_duration = 0.02


def plot_wave(A,B, wave_str):

    t = np.linspace(0,0.12,int(0.12*Fs))

    fig, axs = plt.subplots(2,1,figsize=(12,8),sharex=True)
    fig.suptitle('Test '+wave_str[8:10]+" "+wave_str[-5:-1], fontsize=16)
    
    axs[0].plot(t,A,color="blue")
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('Left Ear')
    axs[0].grid(True)

    axs[1].plot(t,B,color="red")
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('Right Ear')
    axs[1].grid(True)

    plt.show()
    return

def plot_tf(Tes_no,N,T,Tf_matrix,file_path_name, Left):

    xf = fftfreq(N, T)[:N//2]
    

    [TF_Close,TF_OpM,TF_PullL,TF_PullR,TF_Eye] = Tf_matrix

    fig, axs = plt.subplots(5, 2,figsize=(18,16))
    fig.suptitle('Transfer Function Of '+"L "*int(Left)+"R"*int(not(Left))+'-Ear :Test '+Tes_no, fontsize=16)

    axs[0,0].semilogy(xf,(2/N)*abs(TF_Close[:N//2]))
    axs[0,0].set_xlim((F_min,F_max))
    axs[0,0].set_ylim((1e-7,1e-2))
    axs[0,0].set_xlabel('Frequency')
    axs[0,0].set_ylabel('No Expression')
    axs[0,0].grid(True)

    axs[0,1].plot(xf,(180/np.pi)*np.angle(TF_Close[:N//2]))
    axs[0,1].set_xlim((F_min,F_max))
    #axs[0,1].set_ylim((1e-7,1e-2))
    axs[0,1].set_xlabel('Frequency')
    axs[0,1].set_ylabel('Angle(degrees)')
    axs[0,1].grid(True)

    axs[1,0].semilogy(xf,(2/N)*abs(TF_OpM[:N//2]))
    axs[1,0].set_xlim((F_min,F_max))
    axs[1,0].set_ylim((1e-7,1e-2))
    axs[1,0].set_xlabel('Frequency')
    axs[1,0].set_ylabel('Open Mouth')
    axs[1,0].grid(True)

    axs[1,1].plot(xf,(180/np.pi)*np.angle(TF_OpM[:N//2]))
    axs[1,1].set_xlim((F_min,F_max))
    #axs[1,1].set_ylim((1e-7,1e-2))
    axs[1,1].set_xlabel('Frequency')
    axs[1,1].set_ylabel('Angle(degrees)')
    axs[1,1].grid(True)

    axs[2,0].semilogy(xf,(2/N)*abs(TF_PullL[:N//2]))
    axs[2,0].set_xlim((F_min,F_max))
    axs[0,0].set_ylim((1e-7,1e-2))
    axs[2,0].set_xlabel('Frequency')
    axs[2,0].set_ylabel('Pull lip Left')
    axs[2,0].grid(True)

    axs[2,1].plot(xf,(180/np.pi)*np.angle(TF_PullL[:N//2]))
    axs[2,1].set_xlim((F_min,F_max))
    #axs[2,1].set_ylim((1e-7,1e-2))
    axs[2,1].set_xlabel('Frequency')
    axs[2,1].set_ylabel('Angle(degrees)')
    axs[2,1].grid(True)

    axs[3,0].semilogy(xf,(2/N)*abs(TF_PullR[:N//2]))
    axs[3,0].set_xlim((F_min,F_max))
    axs[3,0].set_ylim((1e-7,1e-2))
    axs[3,0].set_xlabel('Frequency')
    axs[3,0].set_ylabel('Pull lip Right')
    axs[3,0].grid(True)

    axs[3,1].plot(xf,(180/np.pi)*np.angle(TF_PullR[:N//2]))
    axs[3,1].set_xlim((F_min,F_max))
    #axs[3,1].set_ylim((1e-7,1e-2))
    axs[3,1].set_xlabel('Frequency')
    axs[3,1].set_ylabel('Angle(degrees)')
    axs[3,1].grid(True)

    axs[4,0].semilogy(xf,(2/N)*abs(TF_Eye[:N//2]))
    axs[4,0].set_xlim((F_min,F_max))
    axs[4,0].set_ylim((1e-7,1e-2))
    axs[4,0].set_xlabel('Frequency')
    axs[4,0].set_ylabel('Eyebrows up')
    axs[4,0].grid(True)

    axs[4,1].plot(xf,(180/np.pi)*np.angle(TF_Eye[:N//2]))
    axs[4,1].set_xlim((F_min,F_max))
    #axs[4,1].set_ylim((1e-7,1e-2))
    axs[4,1].set_xlabel('Frequency')
    axs[4,1].set_ylabel('Angle(degrees)')
    axs[4,1].grid(True)

    plt.savefig(file_path_name)

    #plt.show()

def plot_wave_list(Tes_no,wave_matrix,file_path_name):

    t = np.linspace(0,0.12,int(0.12*Fs))



    fig, axs = plt.subplots(2,5,figsize=(25,6))
    fig.suptitle('Wave forms Test '+Tes_no, fontsize=16)

    axs[0,0].plot(t,wave_matrix[0,0],color="blue")
    axs[0,0].set_ylim((-25,25))
    axs[0,0].set_xlabel('time')
    axs[0,0].set_ylabel('No Expression')
    axs[0,0].grid(True)

    axs[1,0].plot(t,wave_matrix[1,0],color="red")
    axs[1,0].set_ylim((-25,25))
    axs[1,0].set_xlabel('time')
    axs[1,0].set_ylabel('No Expression')
    axs[1,0].grid(True)

    axs[0,1].plot(t,wave_matrix[0,1],color="blue")
    axs[0,1].set_ylim((-25,25))
    axs[0,1].set_xlabel('time')
    axs[0,1].set_ylabel('Open Mouth')
    axs[0,1].grid(True)

    axs[1,1].plot(t,wave_matrix[1,1],color="red")
    axs[1,1].set_ylim((-25,25))
    axs[1,1].set_xlabel('time')
    axs[1,1].set_ylabel('Open Mouth')
    axs[1,1].grid(True)

    axs[0,2].plot(t,wave_matrix[0,2],color="blue")
    axs[0,2].set_ylim((-25,25))
    axs[0,2].set_xlabel('time')
    axs[0,2].set_ylabel('Pull lip Left')
    axs[0,2].grid(True)

    axs[1,2].plot(t,wave_matrix[1,2],color="red")
    axs[1,2].set_ylim((-25,25))
    axs[1,2].set_xlabel('time')
    axs[1,2].set_ylabel('Pull lip Left')
    axs[1,2].grid(True)

    axs[0,3].plot(t,wave_matrix[0,3],color="blue")
    axs[0,3].set_ylim((-25,25))
    axs[0,3].set_ylabel('Side L')
    axs[0,3].set_xlabel('time')
    axs[0,3].set_ylabel('Pull lip Right')
    axs[0,3].grid(True)

    axs[1,3].plot(t,wave_matrix[1,3],color="red")
    axs[1,3].set_ylim((-25,25))
    axs[1,3].set_xlabel('time')
    axs[1,3].set_ylabel('Pull Lip Right')
    axs[1,3].grid(True)

    axs[0,4].plot(t,wave_matrix[0,4],color="blue")
    axs[0,4].set_ylim((-25,25))
    axs[0,4].set_xlabel('time')
    axs[0,4].set_ylabel('Eye-brows up')
    axs[0,4].grid(True)

    axs[1,4].plot(t,wave_matrix[1,4],color="red")
    axs[1,4].set_ylim((-25,25))
    axs[1,4].set_xlabel('time')
    axs[1,4].set_ylabel('Eye-brows up')
    axs[1,4].grid(True)

    plt.savefig(file_path_name)


