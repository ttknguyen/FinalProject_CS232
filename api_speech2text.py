from Models.build_model import CTCLoss
from Utils.preprocessing import encode_single_sample, decode_batch_predictions
from tensorflow import keras
from tensorflow.keras import layers
from jiwer import wer
from bs4 import BeautifulSoup
from gtts import gTTS

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr

import string, glob, os, ffmpeg, scipy, scipy.io.wavfile, IPython



def loadmodel(path):
    model = tf.keras.models.load_model(path, custom_objects={'CTCLoss': CTCLoss})
    return model

def speech2text(model, audio_file, flat = 'our_model'):
    # The set of characters accepted in the transcription.
    lowercase_chars = string.ascii_lowercase
    accented_chars = "àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý"
    final_chars = lowercase_chars + accented_chars  + " "
    characters = [x for x in final_chars]
    # Mapping characters to integers
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    # Mapping integers back to original characters
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    if flat == 'ASR_lib':
        # Use library Speech Recognition
        r = sr.Recognizer()
        my_audio = sr.AudioFile(audio_file)
        with my_audio as source:
            audio = r.record(source)
        result = r.recognize_google(audio, language="vi-VN")

    else:
        # Use our trained model
        spectrogram = np.expand_dims(encode_single_sample(audio_file), axis=0)
        prediction = model.predict(spectrogram)
        model_transcription = decode_batch_predictions(prediction, num_to_char)
        result = model_transcription[0]

    return result
        