import math
from scipy.io import wavfile
import matplotlib.pyplot as plt

e = math.e
pi = math.pi

AUDIO_FILE = "./violin.wav"

class SpectrumAnalyzer:

    def __init__(self):
        self.SAMPLE_RATE = 44100
        plt.style.use('dark_background')

    def exp(self, x : float) -> "Eulers Formula":
        return math.cos(x) + complex(math.sin(x))

    def magnitude(self, real : float, imaginary : complex) -> "Magnitude of a Vector":
        return math.sqrt(real ** 2 + imaginary.real ** 2)

    def DFT(self, samples : list) -> "Discrete Fourier Transform":
        N = len(samples)
        freqBins = []

        for i in range(0, int(N/2)):
            Σ = 0

            for n in range(0, N):
                Σ += samples[n] * self.exp(-(2 * pi * i * n) / N)
            
            freqBins.append(2 * self.magnitude(Σ.real, Σ.imag) / N)

        return freqBins

    def graphResults(self):
        samples = self.loadAudioData()
        freqDomain = self.DFT(samples)
        
        fig, ax = plt.subplots(2, sharex=True)

        fig.suptitle('Discrete Fourier Transform')

        ax[0].plot(samples)
        ax[1].plot(freqDomain)

        ax[0].grid(color='#5a5a5a')
        ax[1].grid(color='#5a5a5a')

        plt.show()

        plt.plot(freqDomain)
        plt.show()

        return self.getStrongestFrequency(freqDomain, samples)

    def getStrongestFrequency(self, frequency_domain, samples):
        return frequency_domain.index(max(frequency_domain)) / len(samples)  * (self.SAMPLE_RATE / 2) 

    def loadAudioData(self):
        self.SAMPLE_RATE, samples = wavfile.read(AUDIO_FILE)
        samples = samples[100000: 101000] # Get first 500 data points

        channel_1 = [channel[0] for channel in samples]
        channel_2 = [channel[1] for channel in samples]

        return channel_1

if __name__ == '__main__':
    dft = SpectrumAnalyzer()
    max_freq = dft.graphResults()

    print("-" * 50)
    print(f"Max frequency: {str(max_freq)} Hz")
    print("-" * 50)
