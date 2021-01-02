# ValuePercept
Examine commonalities between perceptual and value-based processes using functional connectivity data (taken from the [Human Connectome Project]( https://www.humanconnectome.org/study/hcp-young-adult/data-releases)).
OSF link: https://osf.io/bgcxd/

## Code description:
###Congif file (in the code folder):
Include all the directories and paths needed to run the code. Please change it according to your paths.
### Analyze_behavior (in the Behavior folder):
Exploration analyses of the behavior data.<br />
•	Looking specifically at the tasks: Mars contrast (visual task), Flanker task (a higher perceptual task, selective attention), and Delay discount (a value-based task).<br />
•	Z-scoring of the scores for these tasks in order to compare the performance in all tasks 

### PreProcessongZippedFiles (in the code folder):
Pre-processing of the downloaded zipped files. We do this stage once <br />
The data has been downloaded from the denoised resting state data (denoised and aligned across subjects using a functional alignment). <br />
The specific file is ${StudyFolder}/${Subject}/MNINonLinear/Results/${fMRIName}/${fMRIName}_MSMAll_hp2000_clean.dtseries.nii. <br />
•   Unzip the files. <br />	
•	Each subject performed 4 runs. Thus, we have to normalize the signals in each run (to the mean of the run), and then concatenate all the runs for each subject. Then, we create a new file for each subject which contains the 4 concatenated runs: “subject_num.npz”. <br />

### PreProcessingRSdata (in the code folder):
Pre-processing of the resting state fMRI data.<br />
•	Load the concatenated files: each subject had an array of 4800 time points.<br />
•	For each subject: use a specific parcellation which divides the brain into regions of interest (ROIs).<br />
•	Then, create a connectome for each subject: measures the connection between each ROI.<br />
•	For example: if the parcellation is 7 regions of interest, then the connectome would be a 7X7 matrix.<br />
•	Save the matrices, so they can be used in the model code.<br />

### Model_perceptValue:
Initial code for a CPM model- connectome predictive modeling.<br />
•	Load the subjects’ matrices that were produced in the pre-processing code (x)<br />
•	Load behavioral data (y)<br />
•	Equalize the subjects IDs between the two datasets<br />
•	Prepare the matrices (x_features) to the model: take only the upper half of the matrix and without the diagonal (the two halves are the same.)<br />
•	Define CPM functions:<br />
o	Cross validation – working on a leave-one-out instead of k-fold<br />
o	Split train, test<br />
