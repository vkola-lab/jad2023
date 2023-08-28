# Fusion of low-level descriptors of digital voice recordings for dementia assessment

This work is accepted in the _Journal of Alzheimer's Disease_.

## Technical Overview:
The model code is written using Pytorch and this iteration of the code assumes usage of a GPU. The code can be adjusted to run with a CPU instead, although it will be noticeably slower. The data we used from the Framingham Heart Study is not publicly available and so users will have to supply their own data inputs to the model. Users will also have to create their own functions to both read their own data and input their data to the models.

The purpose of this repository is to supply the code that we used to generate our results from using digital voice data from the Framingham Heart Study from our CNN and Random Forest models. However, since we can't supply the actual data we used, the repository can't be directly used to exactly replicate our results - and thus, users must supply their own input data and generate their own results.

## Requirements:
Python 3.6.8 or greater.

Requirements listed in "requirements.txt".

## Installation:
`pip install -r requirements.txt`

## Running the code:
Please run `python binary_train.py -h` to see the various command line arguments that are used to run the CNN model.
Please see `random_forest.py` to see the relevant code for the Random Forest model.

## Files and code that must be supplied by the user:
To reiterate: The data we used from the Framingham Heart Study is not publicly available and so users will have to supply their own data inputs to the model. Users will also have to create their own functions to both read their own data and input their data to the models.
### Task CSV text file
A text file ("task_csvs.txt") that contains the CSV inputs that contain the input data must be supplied by the user. select_task.py shows one way of reading in the CSV path from the task csv text file.
### Select task function [read_text.py]
read_text.py shows how our current code reads our task CSV text file and supplies the CSV input accordingly. The user must supply their own task CSV text file and they can also supply their own select_task() function(s).
## Getting data and labels from the input CSVs [binary_audio_dataset.py]
The initializer for BinaryAudioDataset() contains many attributes that define various functions that are used to read a given input CSV and supply the intended data to the model. The current code defaults all of these various functions to the functions that we used, which are only appropriate to our specific input CSVs (fhs_split_dataframe.py).

Users must change the functions related to querying the input CSVs and getting the label from the input CSVs. The functions related to generating the cross validation folds can likely be reused, but users can adjust them as preferred.

## Viewing model performance
### plot_subplots.py
usage: `python plot_subplots.py <directory_of_cnn_results> <directory_of_cnn_results>, ...`
This script takes in any number of directories of CNN results and will plot ROC-AUC and PR-AUC curves as individual plots, as well as in a single figure with subplots.
