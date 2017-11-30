# Data Pre-processing
Generating spectrograms from input audios (`dataset2spec.py`).

# Part 1 (CS229) - CNN

## Training
Command: `python cnn.py [args]`.
* All arguments have a default value, and it's suggested to double check the values before training, especially `save-dir` and `save-prefix` to avoid overwriting previously trained models. Argument values are printed at the beginning of training.
* Trained models are stored in `output\_models/`.
* Training logs / errs are saved in \*.log/\*.err, with matching names with the trained models.

## Validation
Performance on the validation is checked at the end of each epoch. Please note that validation & testing both use the function 'def validation'; when creating DataLoader, make sure to check the parameters 'shuffle' (False for testing) and 'batch\_size'(*may* want to use batch\_size=1 for testing).

## Testing output of CNN

The saved result matrix is of size `N*109*2` and dimension 3, since there's no time dimension (i.e. 1 image per time):

1. the first dimension is for image id (i.e. N is the number of training image windows); the mapping from image id to annotations follows the code in PitchEstimatorDataSet.py since there was not shuffling during testing;

2. the second dimension is sorted in descending order of probabilities

3. the third dimension is of size 2: position 0 stores the probability, and position 1 stores the corresponding pitch bin.

e.g. mtrx\[0\]\[:\]\[0\] stores in descending order the probabilities for each of the 81 pitch bins at the first timestep of the first song, where the corresponding bin values (between 1-109) are stored in mtrx\[0\]\[:\]\[1\].
