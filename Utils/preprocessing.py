import tensorflow as tf
from tensorflow import keras
import librosa
import numpy as np

# def encode_single_sample(wav_file, frame_length=256, frame_step=160, fft_length=384):
#     # Downsampling the file
#     audio, _ = librosa.load(wav_file, sr=16000)
#     # Change type to float
#     audio = tf.cast(audio, tf.float32)
#     # Get the spectrogram
#     spectrogram = tf.signal.stft(
#         audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
#     )
#     # We only need the magnitude, which can be derived by applying tf.abs
#     spectrogram = tf.abs(spectrogram)
#     spectrogram = tf.math.pow(spectrogram, 0.5)
#     # normalisation
#     means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
#     stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
#     spectrogram = (spectrogram - means) / (stddevs + 1e-10)

#     return spectrogram

def encode_single_sample(wav_file, frame_length=256, frame_step=160, fft_length=384):
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    """
    Describes the transformation that we apply to each element of our dataset
    """
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wav_file)
    # 2. Decode the wav file
    audio, sampling_rate = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    stfts = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(stfts) # get absolute value of complex number
    spectrogram = tf.math.pow(spectrogram, 2) # get power 

    # 6. mel spectrogram
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sampling_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # 8. normalisation
    means = tf.math.reduce_mean(log_mel_spectrograms, 1, keepdims=True)
    stddevs = tf.math.reduce_std(log_mel_spectrograms, 1, keepdims=True)
    log_mel_spectrograms = (log_mel_spectrograms - means) / (stddevs + 1e-10)

    return log_mel_spectrograms

def decode_batch_predictions(pred, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text