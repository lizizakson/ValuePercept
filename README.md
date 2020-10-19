# ValuePercept
Examine commonalities between domains (perception, attention and value) using resting state activity.

## Code description:
### Analyze_behavior:
Exploration analysis of the behavior data.
•	Looking specifically at the tasks: Mars contrast (visual task), Flanker task (a higher perceptual task, selective attention), and Delay discount (a value-based task)./
•	Z-scoring of the scores for these tasks in order to relate between the performances in them

### PreProcessingRSdata:
Pre-processing of the resting state fMRI data.
•	Each subject performed 4 runs. Thus, we have to normalize the signals in each run (to the mean of the run), and then concatenate all the runs for each subject. Then, we create a new file for each subject which contains the 4 concatenated runs: “subject_num.npz”. We do this stage once.
•	Load the concatenated files: each subject had an array of 4800 time points.
•	For each subject: use a specific parcellation which divides the brain into regions of interest (ROIs).
•	Then, create a connectome for each subject: measures the connection between each ROI.
•	For example: if the parcellation is 7 regions of interest, then the connectome would be a 7X7 matrix.
•	Save the matrices, so they can be used in the model code.
•	Some exploration with specific ROIs connections and behavioral scores.

### Model_perceptValue:
Initial code for a CPM model- connectome predictive modeling.
•	Load the subjects’ matrices that were produced in the pre-processing code (x)
•	Load behavioral data (y)
•	Equalize the subjects IDs between the two datasets
•	Prepare the matrices (x_features) to the model: take only the upper half of the matrix and without the diagonal (the two halves are the same.)
•	Define CPM functions:
o	Cross validation – working on a leave-one-out instead of k-fold
o	Split train, test

