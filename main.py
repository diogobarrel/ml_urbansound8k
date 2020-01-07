"""
Machine learning for sound classification attemp
@author: nagihuin
"""

import wav_tools

audio = wav_tools.load_audio_data()
print(audio)
features = wav_tools.load_features_data()
print(features)
