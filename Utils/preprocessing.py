import tensorflow as tf

def encode_single_sample(wav_file, label, char_to_num, 
                            lower_edge_hertz = 80.0, upper_edge_hertz = 7600.0, num_mel_bins = 80, 
                            frame_length = 256, frame_step = 160, fft_length = 384):
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
    ###########################################
    ##  Process the label
    ##########################################
    # 9. Convert label to Lower case
    label = tf.strings.lower(label, encoding="utf-8")
    # 10. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 11. Map the characters in label to numbers
    label = char_to_num(label)
    # 12. Return a dict as our model is expecting two inputs
    return log_mel_spectrograms, label