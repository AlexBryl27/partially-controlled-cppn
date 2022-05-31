import numpy as np
from scipy.io import wavfile


class AudioProcessor:

    
    def __init__(self, track_path, fps=30, wsize=2048):
        
        fs, sound = wavfile.read(track_path)
        self.sound = sound[:, 0]

        self.wsize = wsize
        self.stride = int(fs / fps)


    def condense_spectrum(self, ampspectrum):

        bands = np.zeros(8, dtype=np.float32)

        bands[0] = np.sum(ampspectrum[0:4])
        bands[1] = np.sum(ampspectrum[4:12])
        bands[2] = np.sum(ampspectrum[12:28])
        bands[3] = np.sum(ampspectrum[28:60])
        bands[4] = np.sum(ampspectrum[60:124])
        bands[5] = np.sum(ampspectrum[124:252])
        bands[6] = np.sum(ampspectrum[252:508])
        bands[7] = np.sum(ampspectrum[508:])

        return bands

    
    def get_amplitudes(self, scale=True, scale_rate=0.1, alpha=0.8):
        
        amplitudes = []
        n_samples = len(self.sound)

        for i in range(int(np.ceil(n_samples / self.stride))):
            
            chunk = self.sound[i*self.stride: i*self.stride+self.wsize]
            
            if len(chunk) < self.wsize:
                padsize = self.wsize - len(chunk)
                chunk = np.pad(chunk, (0, padsize), constant_values=0)
                
            freq = np.fft.fft(chunk)[:self.wsize//2]
            amplitudes.append(self.condense_spectrum(np.abs(freq)))
        
        amplitudes = np.stack(amplitudes)

        if scale:
            amplitudes = scale_rate * amplitudes / np.median(amplitudes, axis=0)
        
        if alpha != 1.:
            result = amplitudes.copy()
            for i in range(1, amplitudes.shape[0]):
                for j in range(amplitudes.shape[1]):
                    result[i, j] = alpha * result[i-1, j] + (1 - alpha) * amplitudes[i, j]
            amplitudes = result

        return amplitudes
