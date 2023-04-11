"""File to extract audio embeddings from YamNet."""

import numpy as np
import tensorflow_hub as hub

# imports for testing
from librosa import load

"""Class to extract 1024 audio embeddings from YamNet.
Example:
    >>> from cnn_embeddings import YamNetEmbeddings
    >>> audio = librosa.load("audio_file.wav", sr=16000) # make sure to load audio at 16000 sample rate
    >>> embeddings = YamNetEmbeddings.get_audio_embeddings(audio)
    
    
    Note: Only accepts audio files less than 0.96 seconds long."""

class YamNetEmbeddings:

    YMNET_SF = 16000
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    @staticmethod
    def get_audio_embeddings(waveform):
        waveform = YamNetEmbeddings.__zero_padding(waveform, YamNetEmbeddings.YMNET_SF).astype(np.float32)
        _, embeddings, _ = YamNetEmbeddings.yamnet_model(waveform)

        return embeddings.numpy()
    @staticmethod
    def __zero_padding(audio, sample_rate):
        """Zero pad audio to 0.96 seconds"""
        audio_length = audio.shape[0]
        audio_length = audio_length / sample_rate
        if audio_length < 0.96:
            audio = np.pad(audio, (0, int((0.96 - audio_length)*sample_rate)), 'constant')
        return audio


if __name__ == "__main__":
    # Load yamnet model
    waveform, sample_rate = load("dataset/Sloke/landscape_hold_2023-03-02_21-14-02/Microphone.caf", sr=YamNetEmbeddings.YMNET_SF)
    waveform = waveform[:int(YamNetEmbeddings.YMNET_SF*0.2)]
    embeddings = YamNetEmbeddings.get_audio_embeddings(waveform)

    