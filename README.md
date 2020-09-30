# Audio signal analysis and Feature extraction
Audio signals analysis and feature extraction are important for Automatic Speech Recognition(ASR) task. Audio signals are periodic signals(time-series), hence, we can use RNN models on these signals for ASR. Feature extraction deals with identifying the components that describe the linguistic content and discarding all the other stuff which carry informations like background noise, emotion etc. 

In the notebook we consider a video file(dia1_utt0.mp4) from MELD dataset. Aim is to extract features of the audio signal. The steps involved are
1. Convert video(.mp4) into audio file(.mp3)
2. Perform Time-domain and Frequency domain analysis, to get an overview of the audio signal
3. Extract features(Mel Frequency Cepstral Coefficients a.k.a MFCC) from the audio signal through these steps - 
    1. Break the signal into short frames
    2. Apply windowing function on the frames
    3. Apply Fourier transform on the frames which results in spectrum
    4. Perform Cepstral Analysis to retain the spectral envelope(describes audio). This involves the following steps
       1. Apply the mel filterbank on the spectrum, which results in filter-bank coefficients. Take logarithm of these coefficients 
       2. Take the DCT of the log filter-bank coefficients, which results in MFCCs .Keep 12 MFCCs and discard the rest.
4. Extract features(MFCCs) using librosa package
5. Compare the results of steps 3 and 4

Each step has detailed decription and inference on the results. The resulting MFCCs are given as input to the speech recognition model.

# Acknowledgements
1. https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fn:1
2. http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
3. https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize
4. https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
5. https://archive.org/details/SpectrogramCepstrumAndMel-frequency_636522


