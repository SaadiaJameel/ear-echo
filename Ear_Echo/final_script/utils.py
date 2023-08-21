import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,fftfreq,ifft
from scipy import signal
from scipy.special import rel_entr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import librosa
import csv
import librosa.display

F_min = 17000 # 17KHz
F_max = 23000 # 23KHz
Fs = 100000  # 100KHz
FMCW_duration = 0.08
silent_duration = 0.02
N = 12000
T = 1/Fs

def segment_chirp(recieved_wave,fs, thresh=0.0015, delay= 0.8):
    s = normalize([recieved_wave])[0][int(delay*fs):int(2*fs)]
    chirp_confirm=False
    P=0
    while(chirp_confirm==False):
        key = np.where(s>thresh)[0][0]
        curr = key + int(0.12*fs)
        if (s[curr]>thresh):
            curr = curr+int(0.12*fs)
            if (s[curr]>thresh):
                curr = curr+int(0.12*fs)
                if (s[curr]<thresh):
                    break
                    chirp_confirm=True
                    
        P = P + curr
        s = s[curr:]
        
    P = P + key +  int(delay*fs)-int(0.02*fs)
        
    return( P,P+int(0.46*fs))


def minmax_normalize(data_set):
    data_normalized = (data_set-np.min(data_set))/(np.max(data_set)-np.min(data_set))
    return data_normalized

def standardize(data_set):
    data_normalize = (data_set-np.mean(data_set))/np.std(data_set)
    return data_normalize

def butter_bandpass(lowcut, highcut, fs, order=9):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=9):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs , order=4):
    nyq_freq = fs/2
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y  

def extract_peak_frequency(data, sampling_rate):
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))
    
    peak_coefficient = np.argmax(np.abs(fft_data))
    peak_freq = freqs[peak_coefficient]
    
    return abs(peak_freq * sampling_rate)

def Phase_lag_Hilbert(x1,x2,sampling_rate):
    
    x1h = signal.hilbert(x1)
    x2h = signal.hilbert(x2)
    omega = (extract_peak_frequency(x1,sampling_rate) + extract_peak_frequency(x2,sampling_rate))/2
             
    c = np.inner( x1h, np.conj(x2h) ) / np.sqrt( np.inner(x1h,np.conj(x1h)) * np.inner(x2h,np.conj(x2h)) )
    phase_diff = np.angle(c)/(np.pi*2*omega)
    return(-phase_diff * sampling_rate) # return the delayed number of sample points

def read_data(str_file_name):
    df = pd.read_csv(str_file_name)
    df=df.rename(columns={'Time':'time', 'Channel A':'ChnA', 'Channel B':'ChnB'})
    t = np.array(df.time[1:]).astype(float)
    Chn_A = np.array(df.ChnA[1:]).astype(float)
    Chn_B = np.array(df.ChnB[1:]).astype(float)
   
    return(t/1000 , Chn_A, Chn_B)

def get_data(filename):
    dA_1,dB_1 = np.array([]),np.array([])
    for i in range(1,3):
        s = str(i)
        file_name = filename + ".csv"
        time,A,B = read_data(file_name)
        dA_1,dB_1 = np.concatenate((dA_1,A)),np.concatenate((dB_1,B))
    return(dA_1,dB_1)


def Additionally_average(data_set,fs):
    fs = int(fs)
    averaged_data = data_set[: int(0.120*fs)]

    for i in range(1,3):
     
        averaged_data= averaged_data + data_set[int((0.120*i)*fs) : int((0.120*(i+1))*fs)]
    return(averaged_data.astype(float)/3)

def filtering(array,xf):
    a = np.zeros(len(array))
    for i in range(len(xf)):
        if ((xf[i])>F_min and (xf[i])<F_max):
            a[i]=(array[i]) 

    return (a)

def Dump_CSV(filepath, array):
    # Write arrays to a CSV file
    

    [[TF_Close_L,TF_OpM_L,TF_PullL_L,TF_PullR_L,TF_Eye_L],[TF_Close_R,TF_OpM_R,TF_PullL_R,TF_PullR_R,TF_Eye_R]] = array

    combinedTF_array = np.column_stack((TF_Close_L,TF_Close_R,  
                                            TF_OpM_L, TF_OpM_R,        
                                            TF_PullL_L,TF_PullL_R,
                                            TF_PullR_L, TF_PullR_R,
                                            TF_Eye_L,TF_Eye_R))

    with open(filepath, 'w', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        
        # Write column headers (optional)
        writer.writerow(['RelaxL', 'OpenMouthL', 'lipLeftL', 'lipRightL','EyeL','RelaxR', 'OpenMouthR', 'lipLeftR', 'lipRightR','EyeR'])
        
        # Write array data
        writer.writerows(combinedTF_array)

    # FOR the Transfer Functions # Extrancting important frequency domain data


    return
