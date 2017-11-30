# Automatic Melody Transcription
Automatic melody transcription for multitrack (monophonic) audios. An input audio (`.wav`) is first segmented to short time intervals and transformed to corresponding spectrograms, on which a CNN is run to estimate the probabilities of a pitch present. Finally, with the estimated pitches as inputs, a Hidden Markov Model produces one or more most probable melodies.


# Pipeline

## Part 0 - Data Pre-processing
Generating spectrograms from input audios (`dataset2spec.py`).

## Part 1 (CS229) - CNN for pitch estimation

### Training
Command: `python cnn.py [args]`.
* All arguments have a default value, and it's suggested to double check the values before training, especially `save-dir` and `save-prefix` to avoid overwriting previously trained models. Argument values are printed at the beginning of training.
* Trained models are stored in `output\_models/`.
* Training logs or errors are saved in `*.log / *.err`, with matching names with the trained models.

### Validation
Performance on the validation is checked at the end of each epoch. Please note that validation & testing both use the function 'def validation'; when creating DataLoader, make sure to check the parameters 'shuffle' (False for testing) and 'batch\_size'(*may* want to use batch\_size=1 for testing).

### Testing output of CNN

The saved result matrix is of size `N*109*2` and dimension 3, since there's no time dimension (i.e. 1 image per time):

1. the first dimension is for image id (i.e. N is the number of training image windows); the mapping from image id to annotations follows the code in PitchEstimatorDataSet.py since there was not shuffling during testing;

2. the second dimension is sorted in descending order of probabilities

3. the third dimension is of size 2: position 0 stores the probability, and position 1 stores the corresponding pitch bin.

e.g. mtrx\[0\]\[:\]\[0\] stores in descending order the probabilities for each of the 81 pitch bins at the first timestep of the first song, where the corresponding bin values (between 1-109) are stored in mtrx\[0\]\[:\]\[1\].



## Part 2 (CS221) - HMM for melody tracking
(To do.)


# References
* *MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research* by Bittner, Rachel M and Salamon, Justin and Tierney, Mike and Mauch, Matthias and Cannam, Chris and Bello, Juan Pablo, 2014
* [Deep salience representations for f0 estimation in polyphonic music](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/bittner_deepsalience_ismir_2017.pdf) by Bittner, R and McFee, Brian and Salamon, Justin and Li, Peter and Bello, J., 2017.
* *A Classification Approach to Melody Transcription* by Poliner, Graham E and Ellis, Daniel PW, 2005.
* *Neural network based pitch tracking in very noisy speech* by Han, Kun and Wang, DeLiang, 2014.
* *Convolutional neural network for robust pitch determination* by Su, Hong and Zhang, Hui and Zhang, Xueliang and Gao, Guanglai, ICASSP 2016.
* *HMM-based multipitch tracking for noisy and reverberant speech* by Jin, Zhaozhang and Wang, DeLiang, IEEE 2011.
* *Computer-aided Melody Note Transcription Using the Tony Software: Accuracy and Efficiency* by M. Mauch and C. Cannam and R. Bittner and G. Fazekas and J. Salamon and J. Dai and J. Bello and S. Dixon, 2015.
* *Evaluation of pitch estimation in noisy speech for application in non-intrusive speech quality assessment* by D. Sharma and P. A. Naylor, 2009.
