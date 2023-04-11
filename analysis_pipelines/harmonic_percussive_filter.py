"""Harmonic-percussive separation (HPS) algorithm
Args:
    x (np.ndarray): Time domain input signal

Returns:
    x_h (np.ndarray): Harmonic signal
    x_p (np.ndarray): Percussive signal

Example:
    (tap_h, tap_p) = harmonic_percussive_filter(tap) 

    tap: time series  signal of 200ms (or any duration)
    tap_h: signal with only harmonic sounds
    tap_p: signal with only percussive sounds 
"""

import librosa
import numpy as np
from scipy import signal

def harmonic_percussive_filter(x):

    # these values can be experimented with
    mask = 'soft'
    N = 2048 # window length
    H = 512 # hop size
    eps=0.001 # small epsilon value to avoid division by 0
    L_h = 47 # length of horizontal filter
    L_p = 9 # length of vertical filter

    # stft - convert time domain to frequency domain
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann', center=True, pad_mode='constant')
    # power spectrogram
    Y = np.abs(X) ** 2
    
    # median filtering

    Y_h = signal.medfilt(Y, [1, L_h])
    Y_p = signal.medfilt(Y, [L_p, 1])

    # masking
    if mask == 'binary':
        M_h = np.int8(Y_h >= Y_p)
        M_p = np.int8(Y_h < Y_p)
    if mask == 'soft':
        eps = 0.00001
        M_h = (Y_h + eps / 2) / (Y_h + Y_p + eps)
        M_p = (Y_p + eps / 2) / (Y_h + Y_p + eps)
    X_h = X * M_h
    X_p = X * M_p

    # istft - convert frequency domain back to time domain
    x_h = librosa.istft(X_h, hop_length=H, win_length=N, window='hann', center=True, length=x.size)
    x_p = librosa.istft(X_p, hop_length=H, win_length=N, window='hann', center=True, length=x.size)

    return (x_h, x_p)