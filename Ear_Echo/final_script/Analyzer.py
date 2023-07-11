import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils , ploter
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


def pre_process(data_file_name , Inf_signal, plot=False):

    plot_name = data_file_name[8:]
    Amp_thresh , time_shift =0.0013, 0.1

    A_data , B_data = utils.get_data(data_file_name)
    
    Found = True

    while(Found):
        try:
            print(data_file_name)
            ind = utils.segment_chirp(A_data,Fs,Amp_thresh,time_shift)
            Chirp_signal_A = ([A_data])[0][ind[0]:ind[1]]
            Chirp_signal_B = ([B_data])[0][ind[0]:ind[1]]

            Pulse_A = utils.Additionally_average(Chirp_signal_A, Fs)
            Pulse_B = utils.Additionally_average(Chirp_signal_B , Fs)

            Pulse_A = utils.butter_bandpass_filter(Pulse_A, F_min, F_max, Fs,order=9)
            Pulse_B = utils.butter_bandpass_filter(Pulse_B, F_min, F_max, Fs,order=9)



            if (abs(np.dot(Inf_signal,Pulse_A))>12000): Found = False
            else:
                time_shift = time_shift + 0.1
                if time_shift>1.0 : time_shift = 0.1
                
        except:
                time_shift = time_shift + 0.05
                if time_shift>1.1 : time_shift = 0.1
            
            
    
    if (plot):
        ploter.plot_wave(Pulse_A,Pulse_B,plot_name)
    

    return (Pulse_A , Pulse_B)


##########################################################################
Intfer_A_d,Intfer_B_d = utils.get_data("Data\Infer")
ind = utils.segment_chirp(Intfer_A_d,Fs,thresh=0.0013,delay=0.6)
Intfer_A = ([Intfer_A_d])[0][ind[0]:ind[1]]
Intfer_B = ([Intfer_B_d])[0][ind[0]:ind[1]]

Inf_A =    utils.butter_bandpass_filter(utils.Additionally_average(Intfer_A,Fs), F_min, F_max, Fs,order=9)
Inf_B =   utils.butter_bandpass_filter(utils.Additionally_average(Intfer_B,Fs), F_min, F_max, Fs,order=9)

#########################################################################
ref_A0d , ref_B0d = utils.get_data("Data\Tes0\Ref")
ind = utils.segment_chirp(ref_A0d,Fs,thresh=0.0013,delay=0.4)
ref_A0 = ([ref_A0d])[0][ind[0]:ind[1]]
ref_B0 = ([ref_B0d])[0][ind[0]:ind[1]]

Base_refA = utils.butter_bandpass_filter(utils.Additionally_average(ref_A0,Fs), F_min, F_max, Fs,order=9)
Base_refB = utils.butter_bandpass_filter(utils.Additionally_average(ref_B0,Fs), F_min, F_max, Fs,order=9)

##########################################################################
T= 1/Fs
N=len(Base_refA)
X_f = fftfreq(N,T)

F_interferance_A , F_interferance_B = fft(Inf_A) , fft(Inf_B)
F_ref_A0 , F_ref_B0 = fft(Base_refA) , fft(Base_refB)

def get_TF(signalA,signalB, ref_A, ref_B ):
    
    signalA_fft , signalB_fft = fft(signalA) , fft(signalB)
    F_ref_A , F_ref_B = fft(ref_A) , fft(ref_B)

    A , B = F_ref_A0/F_ref_A , F_ref_B0/F_ref_B
    
    tfA= ((signalA_fft*A)-F_interferance_A)/ F_interferance_A

    tfB = ((signalB_fft*B)-F_interferance_B)/ F_interferance_B

    tf_A_dB = 10*np.log10(abs(tfA))
    tf_B_dB = 10*np.log10(abs(tfB))

    TFA = utils.filtering(tf_A_dB , X_f)
    TFB = utils.filtering(tf_B_dB , X_f)

    return(TFA,TFB)




for i in range(1):

    Tes_no = str(i)
    Ref_A,Ref_B = pre_process("Data\Tes"+Tes_no+"\Ref" ,Base_refA,plot=False )
    Relax_A,Relax_B = pre_process("Data\Tes"+Tes_no+"\Close",Base_refA,plot=False )
    OpM_A,OpM_B = pre_process("Data\Tes"+Tes_no+"\Open",Base_refA,plot=False )
    PullL_A,PullL_B = pre_process("Data\Tes"+Tes_no+"\PullL",Base_refA,plot=False )
    PullR_A , PullR_B = pre_process("Data\Tes"+Tes_no+"\PullR",Base_refA,plot=False )
    EyeUp_A , EyeUp_B = pre_process("Data\Tes"+Tes_no+"\Eye",Base_refA,plot=False )

    TF_Close_L , TF_Close_R = get_TF(Relax_A,Relax_B ,Ref_A,Ref_B )
    TF_OpM_L , TF_OpM_R = get_TF(OpM_A,OpM_B ,Ref_A,Ref_B )
    TF_PullL_L , TF_PullL_R = get_TF(PullL_A,PullL_B ,Ref_A,Ref_B )
    TF_PullR_L , TF_PullR_R = get_TF(PullR_A , PullR_B  ,Ref_A,Ref_B )
    TF_Eye_L , TF_Eye_R = get_TF(EyeUp_A , EyeUp_B ,Ref_A,Ref_B )
    
    lower_cut = int(len(X_f)/2 * (F_min)/max(X_f)) + 1
    upper_cut = int(len(X_f)/2 * (F_max)/max(X_f)) + 1

    

    TF_Matrix = np.array([[TF_Close_L[lower_cut:upper_cut] ,TF_OpM_L[lower_cut:upper_cut],TF_PullL_L[lower_cut:upper_cut],TF_PullR_L[lower_cut:upper_cut],TF_Eye_L[lower_cut:upper_cut]],
                            [TF_Close_R[lower_cut:upper_cut],TF_OpM_R[lower_cut:upper_cut],TF_PullL_R[lower_cut:upper_cut],TF_PullR_R[lower_cut:upper_cut],TF_Eye_R[lower_cut:upper_cut]]])
    
    #ploter.plot_tf(Tes_no, N ,T , TF_Matrix[0],True)

    #ploter.plot_tf(Tes_no, N ,T , TF_Matrix[1],False)


    utils.Dump_CSV(Tes_no,TF_Matrix)

