# Automatic Melody Transcription
Automatic melody transcription for multitrack (monophonic) audios. An input audio (`.wav`) is first segmented to short time intervals and transformed to corresponding spectrograms, on which a CNN is run to estimate the probabilities of a pitch present. Finally, with the estimated pitches as inputs, a Hidden Markov Model produces one or more most probable melodies.

# Main files
* `librosa_baseline.py`: baseline using the `librosa` library.
* `PitchEstimationDataSet.py`: for loading dataset.
* CNNs: `cnn.py` is the plain CNN; `cnn_fusion.py` is for early/late fusion; `cnn_bin.py` and `cnn_multi.py` are still in progress.
  * Use `config.py` to specify your configuration.
  * Models are specified under `model/`
* LSTM: `lstm_eval.py` is for training the LSTM; `lstm_test.py` is for testing.
* HMM: `eval_hmm.py` is the plain HMM on the entire song; `eval_hmm_fragmented.py` applies HMM no various lengths of audio segments; `eval_hmm_laplacian.py` was an early experiment using the Laplacian model.
* Error analysis:
  * `feature_visualisation.py`: PCA + t-SNE
  * `plot_cnf.py`: confusion matrix
* Util functions:
  * `util.py`: various util functions; there is a list of functions at the beginning of the file
  * `util_cnn.py`: util functions for CNN
  * MIDI file: `generate_midi_files.py`


# Dependencies 
The project depends on the following packages 
- PyTorch
- Matplotlib
- MIDIFile
- Librosa 

# Pipeline

## Part 0 - Data Pre-processing
Generating spectrograms from input audios (`dataset2spec.py`).
Note that the sampling rate of the spectrograms is half the rate of the annotations; i.e. we should skip every other line in the annotations (e.g. refer to `util.py/read_melody` where we keep track of `count % sr_ratio`)
* Data under subdirectories of `/root/new_data/`. In each subdirectory, `annot/` for annotations, `audios/` for `wav` files, and `image` for spectrogram slices grouped by songs.


![spec_whole_song](https://user-images.githubusercontent.com/13089230/33816412-8d4730f0-dded-11e7-9a76-ad394671f6d8.jpg)
Example of spectrogram generated from an audio file 

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

e.g. mtrx\[0\]\[:\]\[0\] stores in descending order the probabilities for each of the 109 pitch bins at the first timestep of the first song, where the corresponding bin values (between 0-108, 0 being empty pitch) are stored in mtrx\[0\]\[:\]\[1\].

### Error Analysis 
* Confusion matrix: `python confusion_matrix.py` generates the confusion matrix on the validation data set and saves it to `cnf_matrix.npy`. 

![68567649](https://user-images.githubusercontent.com/13089230/33816426-ab8f8ce2-dded-11e7-86ea-b4c9ea41d79b.png)
Sample confusion matrix on the validation set 

* Features visualisation: `python features_visualisation.py` performs PCA  and t-SNE on the 8192-dimension feature vector output from the CNN before the fully connected layers in order to embed the vectors in 2D space for visualisation. 

![features2d-3](https://user-images.githubusercontent.com/13089230/33816396-68f746cc-dded-11e7-8d80-c6757b166e85.png)
Visualisation of some features categorized by bins

## Part 2 (CS221) - HMM for melody tracking

### Training the transition probabilities 
The transition probabilities are estimated on the training data set using smoothed maximum likelihood estimates of the probability of transitioning from bin i to bin j. To launch the training 
```
python train_mle.py
```
This will perform the MLE counting on the annotations for the training set and save the probabilities in `transitions_mle.npy` as a dictionary. 

### Implementation details
Pitch tracking is modeled as Hidden Markov Model that we solve using Gibbs Sampling. The relevant code for the underlying CSP structure is in `csp.py`. The starter code was provided as part of a CS221 assignment that we complemented with our own implementation of backtracking search and gibbs sampling to find an optimal assignment. 

`Pitch_Contour.py` extends the CSP class. Custom emission and transition probabilities for the pitch tracking problem can be set using `set<Emission|Start|Transition>Probability()`. By default, the emission probability is distributed following a multinomial, the transition probability is distributed following a laplacian distribution and the start probability is uniformly distributed.

Calling `solve()` computes an optimal assignment. The default method is Gibbs sampling. To use backtracking use `solve(mode='backtrack')`. 

### Evaluating the HMM
To evaluate the HMM, we load the results of the CNN on the validation set and perform pitch tracking on each song of the set through Gibbs Sampling. The result is then compared time frame by time frame to the annotations.
Evaluation can be launched using 
```
python eval_hmm.py
``` 
By default this launches the evaluation for an HMM built using the learned transition probabilities and the multinomial model on each hidden variable and outputs the total accuracy on the validation set. This also separates the initial input of the HMM in two different matrices depending on whether the proposed value matches the ground truth and saves them so that they can be reused for error analysis.


### Error Analysis 
In order to understand in which cases the HMM seem to be improving the results given from the CNN, we analyse the influence of the top-1 and top-2 probabilities given by the CNN's softmax layer as well as the influence of the proposed bin values on the classification result. 
To see the repartition of top-1 probabilities depending on positive / negative results:  
``` 
python error_analysis_hmm.py
``` 

![first-probabilities-refined-step50](https://user-images.githubusercontent.com/13089230/33816410-8824f076-dded-11e7-80f2-cd0976378c13.png)
Histogram of the repartition for the top-1 probabilities provided by the CNN. The repartition between bad and positive results is mostly similar until a probability threshold after which they diverge. 

## Part 3 - Inference
To test the entire pipeline, 
``` 
python inference.py
```
This will run the entire inference process on a given song by computing the spectogram images, saving them in a directory, then performing pitch estimation on each time frame using the pre-trained CNN and finally perform pitch-tracking. The result will then be converted into a MIDI file. 

Generating MIDI files 
MIDI files for the results on the dataset can be generated using 
```
python generate_midi_files.py
```
This will generate MIDI files from the dataset annotations and the inference result of the models and save them. The results can then be visualised and compared using a software like Musescore. Note that since the tempo was not given from the dataset, it is arbitrarily defined and we rely solely on the timestamps given by the annotations to model the duration of a note. 
The script needs to be given the following in order to work 
- location of the original annotations.
- location of the result matrix giving the softmax output of the CNN on each of the songs that needs to be visualised. This needs to be in the same format specified in the CNN section (Testing output)
- location of the `.npy` file containing the trained Maximum Likelihood transition estimates 


![schumann_mignon_original](https://user-images.githubusercontent.com/13089230/33816438-be90b46a-dded-11e7-8f9d-c0f2ce635328.png)

MIDI file generated from annotations (Schumann Mignon)

![schumann_mignon_inference_065](https://user-images.githubusercontent.com/13089230/33816437-be7992e4-dded-11e7-9790-82f5200999e0.png)

MIDI file generated from our inference pipeline for the same audio input 

# Documentation
* [Milestone](https://www.overleaf.com/12132890syqfthdckmgj#/46086031/)
* Final Report (to be released)
  * [Google spreadsheet](https://docs.google.com/spreadsheets/d/1KYRvSyM2JVV2ZiFddUcCX4tu9Kxq0nLXLjpy8P8qIZE/edit#gid=0) for HMM results
* [CS221 Poster](https://drive.google.com/open?id=1_yV6NLu-qXVhDv-ff0-_BnmhvfIXAT-s)

# References
* **MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research** by Bittner, Rachel M and Salamon, Justin and Tierney, Mike and Mauch, Matthias and Cannam, Chris and Bello, Juan Pablo, 2014
* **[Deep salience representations for f0 estimation in polyphonic music](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/bittner_deepsalience_ismir_2017.pdf)** by Bittner, R and McFee, Brian and Salamon, Justin and Li, Peter and Bello, J., 2017.
* **A Classification Approach to Melody Transcription** by Poliner, Graham E and Ellis, Daniel PW, 2005.
* **Neural network based pitch tracking in very noisy speech** by Han, Kun and Wang, DeLiang, 2014.
* **Convolutional neural network for robust pitch determination** by Su, Hong and Zhang, Hui and Zhang, Xueliang and Gao, Guanglai, ICASSP 2016.
* **HMM-based multipitch tracking for noisy and reverberant speech** by Jin, Zhaozhang and Wang, DeLiang, IEEE 2011.
* **Computer-aided Melody Note Transcription Using the Tony Software: Accuracy and Efficiency** by M. Mauch and C. Cannam and R. Bittner and G. Fazekas and J. Salamon and J. Dai and J. Bello and S. Dixon, 2015.
* **Evaluation of pitch estimation in noisy speech for application in non-intrusive speech quality assessment** by D. Sharma and P. A. Naylor, 2009.

# To-Do
* Unbalanced class: 
  * treat as a detection problem: still training; code in `cnn_bin.py`
  * add weight to loss function
* Multiclass / polyphonic
  * training using `cnn_multi.py`
  * running on conv5 (experiment with the smaller model since the performance gain from conv7 is small)
* Data augmentation
  * adding noise
  * changing volume: not applicable since it is normalized
* More features
  * max spacing to help determine foundamental frequency

## Log for finished to-dos
* Longer context for data (46ms)
  * more data: 34 -> 60 songs
* Use Q-transformation for more features: 1.7% increase as for the first epoch
* Deeper network
  * 7 conv layers + same FCs: ~2x model size (3574 vs 7561); slightly better results.
* LSTM for part 2
  * `lstm.py`: main file / entry point. Command: `python lstm.py >your_log_file`
  * `model_lstm.py`: specifies the network structure. Currently using 2 hidden layers w/ 1024 nodes each.
  * `LSTMDataset.py`: loads the data s.t. data/target from `__getitem__()` are of size `batch_size * seq_len`. Currently the sequence length is 1000.
  * **Note**: curently the training loss is not decreasing. **Possible reasons** include: buggy implementation; suboptimal hyperparameters; inappropriate sequence length for LSTM.


