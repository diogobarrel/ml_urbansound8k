"""
Class created to handle wav files
"""
import os
import struct

import wave
import simpleaudio

import pandas
import numpy

import matplotlib.pyplot as plt

from host.features import metadata, urbansound8k_audio, extract_features
from utils import timeit


class WavFileHelper():
    """
    Main class created to manage audio: read sound file proprieties, play sound
    and transform it in digital signal
 
    """
    def read_file_props(self, file):
        """
        read file
        """
        wave_file = open(file, "rb")

        riff = wave_file.read(12)
        fmt = wave_file.read(36)

        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]

        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I", sample_rate_string)[0]

        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H", bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)

    def signal(self, filename, sample_frequency=16000):
        """ draw signal wave file and sectrum of signal """
        signal_wave = wave.open(filename, 'r')

        data = numpy.fromstring(signal_wave.readframes(sample_frequency),
                                dtype=numpy.int16)

        sig = signal_wave.readframes(-1)
        sig = numpy.fromstring(sig, 'Int16')
        return sig


    @timeit
    def plot_wave_spectrum(self, signals, plots=1):
        """ plot wave and spectrum of the signal wave file """
        if signals and not isinstance(signals, list):
            sig = signals
            plt.figure(plots)

            a = plt.subplot(211)
            a.set_xlabel('time [s]')
            a.set_ylabel('sample value [-]')
            plt.plot(sig)

            c = plt.subplot(212)
            Pxx, freqs, bins, im = c.specgram(sig,
                                            NFFT=1024,
                                            Fs=16000,
                                            noverlap=900)
            c.set_xlabel('Time')
            c.set_ylabel('Frequency')
            plt.show()

        if signals:
            samples = list(range(len(signals)))

            for sample, sig in list(zip(samples, signals)):
                plt.figure(sample)
                a = plt.subplot(211)
                a.set_xlabel('time [s]')
                a.set_ylabel('sample value [-]')
                plt.plot(sig)

                c = plt.subplot(212)
                Pxx, freqs, bins, im = c.specgram(sig,
                                                NFFT=1024,
                                                Fs=16000,
                                                noverlap=900)
                c.set_xlabel('Time')
                c.set_ylabel('Frequency')

            plt.show()


@timeit
def load_audio_data():
    audiodata = []
    wavfilehelper = WavFileHelper()

    for index, row in metadata.iterrows():

        file_name = os.path.join(os.path.abspath(urbansound8k_audio),
                                 'fold' + str(row["fold"]) + '/',
                                 str(row["slice_file_name"]))

        data = wavfilehelper.read_file_props(file_name)
        audiodata.append(data)

    # Convert into a Panda dataframe
    audiodf = pandas.DataFrame(
        audiodata, columns=['num_channels', 'sample_rate', 'bit_depth'])

    return audiodf


@timeit
def load_features_data():
    """Load files in a list of paths and extract their audio features
    
    
    Returns:
        features_df: pandas.Dataframe
    """    
    features = []
    wavfilehelper = WavFileHelper()

    # Iterate through each sound file and extract the features
    for index, row in metadata.iterrows():

        file_name = os.path.join(os.path.abspath(urbansound8k_audio),
                                 'fold' + str(row["fold"]) + '/',
                                 str(row["slice_file_name"]))

        class_label = row["class"]
        data = extract_features(file_name)

        features.append([data, class_label])

    # Convert into a Panda dataframe
    featuresdf = pandas.DataFrame(features, columns=['feature', 'class_label'])

    return featuresdf


@timeit
def play_sound(self, file_path):
    """
    WAV files contain a sequence of bits representing the raw audio data,
    as well as headers with metadata in RIFF (Resource Interchange File Format)
    format.

    For CD recordings, the industry standard is to store each audio
    sample (an individual audio datapoint relating to air pressure) as a
    16-bit value, at 44100 samples per second.

    To reduce file size, it may be sufficient to store some recordings
    (for example of human speech) at a lower sampling rate, such as
    8000 samples per second, although this does mean that higher sound
    frequencies may not be as accurately represented.

    A few of the libraries discussed in this tutorial play and record bytes objects
    whereas others use NumPy arrays to store raw audio data. 

    @realpython
    """
    if not file_path:
        return

    wave_obj = simpleaudio.WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait untill the sound has finished playing


@timeit
def pitch():
    """
    creates and play a random pitch
    """
    frequency = 440  # Our played note will be 440 Hz
    fs = 44100  # 44100 samples per second
    seconds = 3  # Note duration of 3 seconds

    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = numpy.linspace(0, seconds, seconds * fs, False)

    # Generate a 440 Hz sine wave
    note = numpy.sin(frequency * t * 2 * numpy.pi)

    # Ensure that highest value is in 16-bit range
    audio = note * (2**15 - 1) / numpy.max(numpy.abs(note))
    # Convert to 16-bit data
    audio = audio.astype(numpy.int16)

    # Start playback
    play_obj = simpleaudio.play_buffer(audio, 1, 2, fs)

    # Wait for playback to finish before exiting
    play_obj.wait_done()
