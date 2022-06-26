import pandas as pd
import numpy as np
import tensorflow as tf
import string
import glob
import librosa

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer
import os

def loaddataset():
    # cd to the folder contains file .tar.gz of the dataset
    data_path = "data/vivos"
    train_data_path = os.path.join(data_path, "train")
    test_data_path = os.path.join(data_path, "test")

    # Read metadata file and parse it
    train_prompts_path = os.path.join(train_data_path, "prompts.txt")
    test_prompts_path = os.path.join(test_data_path, "prompts.txt")

    train_wavs_path = os.path.join(train_data_path, "waves")
    test_wavs_path = os.path.join(test_data_path, "waves")

    with open(train_prompts_path, encoding="utf8") as f:
        data = f.read()
        lines = data.split("\n")
        train_file_path = []
        train_transcription = []
        for line in lines:
            instance = line.split(" ", 1)
            if len(instance) == 1:
                continue
            file_path = glob.glob(os.path.join(train_wavs_path, "**", instance[0] + ".wav"))[0]
            train_file_path.append(file_path)
            train_transcription.append(instance[1])
        d = {"file_path": train_file_path, "transcription": train_transcription}
        train_df = pd.DataFrame(d)

    with open(test_prompts_path, encoding="utf8") as f:
        data = f.read()
        lines = data.split("\n")
        test_file_path = []
        test_transcription = []
        for line in lines:
            instance = line.split(" ", 1)
            if len(instance) == 1:
                continue
            file_path = glob.glob(os.path.join(test_wavs_path, "**", instance[0] + ".wav"))[0]
            test_file_path.append(file_path)
            test_transcription.append(instance[1])
        d = {"file_path": test_file_path, "transcription": test_transcription}
        test_df = pd.DataFrame(d)

    return train_df, test_df

