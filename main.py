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





    



    




'''from fft import * # Custom package import
import math
from scipy.io import wavfile
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

AUDIO_FILE = "./violin.wav"

fs, data = wavfile.read(AUDIO_FILE)

audio_channel_1 = [channel[0] for channel in data]
audio_channel_2 = [channel[1] for channel in data]

samples = audio_channel_1[100000: 101000] # Get first 500 data points

plt.plot(samples)
plt.show()

dft_result = DFT(samples)

plt.plot(dft_result)
plt.show()

max_freq = dft_result.index(max(dft_result)) / len(samples)  * (fs / 2) 
dft_result.pop(dft_result.index(max(dft_result)))
max_second_freq = dft_result.index(max(dft_result)) / len(samples)  * (fs / 2) 

print("-" * 50)
print(f"Max frequency: {str(max_freq)} Hz")
print("-" * 50)

import numpy as np
import pyaudio

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class SpectrumAnalyzer:
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    CHUNK = 512
    START = 0
    N = 512

    wave_x = 0
    wave_y = 0
    spec_x = 0
    spec_y = 0
    data = []

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format = self.FORMAT,
            channels = self.CHANNELS, 
            rate = self.RATE, 
            input = True,
            output = False,
            frames_per_buffer = self.CHUNK)
        # Main loop
        self.loop()

    def loop(self):
        try:
            while True :
                self.data = self.audioinput()
                self.fft()
                self.graphplot()

        except KeyboardInterrupt:
            self.pa.close()

    def audioinput(self):
        ret = self.stream.read(self.CHUNK, exception_on_overflow=False)
        ret = np.fromstring(ret, np.float32)
        return ret

    def fft(self):
        self.wave_x = range(self.START, self.START + self.N)
        self.wave_y = self.data[self.START:self.START + self.N]
        self.spec_x = np.fft.fftfreq(self.N, d = 1.0 / self.RATE)  
        y = np.fft.fft(self.data[self.START:self.START + self.N])    
        self.spec_y = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in y]

    def graphplot(self):
        plt.clf()
        # wave
        plt.subplot(311)
        plt.plot(self.wave_x, self.wave_y)
        plt.axis([self.START, self.START + self.N, -0.5, 0.5])
        plt.xlabel("time [sample]")
        plt.ylabel("amplitude")
        #Spectrum
        plt.subplot(312)
        plt.plot(self.spec_x, self.spec_y, marker= 'o', linestyle='-')
        plt.axis([0, self.RATE / 2, 0, 50])
        plt.xlabel("frequency [Hz]")
        plt.ylabel("amplitude spectrum")
        #Pause
        plt.pause(.01)

if __name__ == "__main__":
    spec = SpectrumAnalyzer()'''

'''import matplotlib
matplotlib.use('TkAgg') # THIS MAKES IT FAST!
import numpy
import scipy
import struct
import pyaudio
import threading
import pylab
import struct

class SwhRecorder:
    """Simple, cross-platform class to record from the microphone."""

    def __init__(self):
        """minimal garb is executed when class is loaded."""
        self.RATE=48100
        self.BUFFERSIZE=2**12 #1024 is a good buffer size
        self.secToRecord=.1
        self.threadsDieNow=False
        self.newAudio=False

    def setup(self):
        """initialize sound card."""
        #TODO - windows detection vs. alsa or something for linux
        #TODO - try/except for sound card selection/initiation

        self.buffersToRecord=int(self.RATE*self.secToRecord/self.BUFFERSIZE)
        if self.buffersToRecord==0: self.buffersToRecord=1
        self.samplesToRecord=int(self.BUFFERSIZE*self.buffersToRecord)
        self.chunksToRecord=int(self.samplesToRecord/self.BUFFERSIZE)
        self.secPerPoint=1.0/self.RATE

        self.p = pyaudio.PyAudio()
        self.inStream = self.p.open(format=pyaudio.paInt16,channels=1,
            rate=self.RATE,input=True,frames_per_buffer=self.BUFFERSIZE)
        self.xsBuffer=numpy.arange(self.BUFFERSIZE)*self.secPerPoint
        self.xs=numpy.arange(self.chunksToRecord*self.BUFFERSIZE)*self.secPerPoint
        self.audio=numpy.empty((self.chunksToRecord*self.BUFFERSIZE),dtype=numpy.int16)

    def close(self):
        """cleanly back out and release sound card."""
        self.p.close(self.inStream)

    ### RECORDING AUDIO ###

    def getAudio(self):
        """get a single buffer size worth of audio."""
        audioString=self.inStream.read(self.BUFFERSIZE)
        return numpy.fromstring(audioString,dtype=numpy.int16)

    def record(self,forever=True):
        """record secToRecord seconds of audio."""
        while True:
            if self.threadsDieNow: break
            for i in range(self.chunksToRecord):
                self.audio[i*self.BUFFERSIZE:(i+1)*self.BUFFERSIZE]=self.getAudio()
            self.newAudio=True
            if forever==False: break

    def continuousStart(self):
        """CALL THIS to start running forever."""
        self.t = threading.Thread(target=self.record)
        self.t.start()

    def continuousEnd(self):
        """shut down continuous recording."""
        self.threadsDieNow=True

    ### MATH ###

    def downsample(self,data,mult):
        """Given 1D data, return the binned average."""
        overhang=len(data)%mult
        if overhang: data=data[:-overhang]
        data=numpy.reshape(data,(len(data)/mult,mult))
        data=numpy.average(data,1)
        return data

    def fft(self,data=None,trimBy=10,logScale=False,divBy=100):
        if data==None:
            data=self.audio.flatten()
        left,right=numpy.split(numpy.abs(numpy.fft.fft(data)),2)
        ys=numpy.add(left,right[::-1])
        if logScale:
            ys=numpy.multiply(20,numpy.log10(ys))
        xs=numpy.arange(self.BUFFERSIZE/2,dtype=float)
        if trimBy:
            i=int((self.BUFFERSIZE/2)/trimBy)
            ys=ys[:i]
            xs=xs[:i]*self.RATE/self.BUFFERSIZE
        if divBy:
            ys=ys/float(divBy)
        return xs,ys

    ### VISUALIZATION ###

    def plotAudio(self):
        """open a matplotlib popup window showing audio data."""
        pylab.plot(self.audio.flatten())
        pylab.show()

if __name__ == "main":
    fft = SwhRecorder()
    fft.setup()
    fft.plotAudio()
    fft.continuousStart()'''
