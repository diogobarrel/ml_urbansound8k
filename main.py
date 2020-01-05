# Load various imports
import os
import pandas

from pprint import pprint

from host.features import metadata, urbansound8k_audio
from wav_tools import WavFileHelper
"""
starts wavfilehelper to load data into panda dataframes
"""
wavfilehelper = WavFileHelper()
audiodata = []
for index, row in metadata.iterrows():

    file_name = os.path.join(os.path.abspath(urbansound8k_audio),
                             'fold' + str(row["fold"]) + '/',
                             str(row["slice_file_name"]))

    data = wavfilehelper.read_file_props(file_name)
    audiodata.append(data)

# Convert into a Panda dataframe
pprint(audiodata)
audiodf = pandas.DataFrame(audiodata,
                       columns=['num_channels', 'sample_rate', 'bit_depth'])
pprint(audiodf.values)