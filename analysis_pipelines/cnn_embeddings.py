"""File to extract audio embeddings from a CNN model."""

import torch
import torchaudio
import numpy as np

import tensorflow_hub as hub

import torch
import torchaudio
from librosa import load


YMNET_SF = 16000


def get_audio_embeddings(audio_data, yamnet_model):
    # Load audio data using torchaudio
    # import pdb; pdb.set_trace()
    waveform, sample_rate = torchaudio.load(audio_data)
    # use librosa load instead and convert to tensor

    
    import pdb; pdb.set_trace()
    # resample to 16kHz
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=YMNET_SF)(waveform)


    # Preprocess audio data for input to the YAMNet model
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = zero_padding(waveform, YMNET_SF)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=YMNET_SF, n_fft=2048, hop_length=512, n_mels=64)(waveform)
    mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    input_data = torch.transpose(mel_spectrogram, 1, 2)

    # Convert audio data to a tensor and apply YAMNet model
    with torch.no_grad():
        audio_embedding = yamnet_model(input_data)

    return audio_embedding

def zero_padding(audio, sample_rate):
    """Zero pad audio to 0.96 seconds"""
    audio_length = audio.shape[1]
    audio_length = audio_length / sample_rate
    if audio_length < 0.96:
        audio = np.pad(audio, (0, int(0.96 * sample_rate) - audio_length), 'constant')
    return audio

if __name__ == "__main__":
    # Load yamnet model
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    audio_embedding = get_audio_embeddings("/Users/sloke"\
                                            "/Downloads/Microphone.wav", yamnet_model)

    