# BrainConnectomeDeepLearning
These are the code used for my paper: Predicting sex, age, general cognition and mental health with machine learning on brain structural connectomes https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.26182

For detail decription of the BrainNetCNN model, please refer to Jeremy Kawahara et al. https://www.sciencedirect.com/science/article/abs/pii/S1053811916305237

The DLGraphAndLayers folder contains the DAGnet construction and custom layers. For BrainNetCNN model without external regressor, one can also use the 
BrainNetCNNModel_resnet.m matlab function or the BrainNetCNNModelBuilding.py python function

## Analysis
Suppose we have the a .mat file that contains all the Data needed for the analysis

In the Data.mat file of N = 8183 participants, we have variables:

- ID 		- 	Participant ID
- Connectome	-	Normalized connectomes (normalization described in the paper), struct variable, 
			with fields {MD, FA, SC, OD, ISOVF, ICVF}, each of these weighted connectomes are of dimension 85-by-85-by-8183
  - SC: uncorrected streamline count
  - FA: fractional anisotropy
  - MD: mean diffusivity
  - ICVF: intracellular volume fraction (neurite density)
  - ISOVF: isotropic volume fraction (extracellular water diffusion)
  - OD: orientation dispersion (tract fanning/complexity)
- TargetVariables - An array ID, Age, Sex, g-factor, MHQ-factor, nan for missing (for derivation of g-factor and MHQ-factor, please refer to the paper)


Model Training and collecting results:

First, add all the paths 
  
Run DL_AllPredictionTasks_SavingTrainingProgress.m to get all the BrainNetCNNModel results, the core parts under this script are:

1. BrainNetCNNModelTraining.m	- Model training as well as saving the checkpoint networks at each epoch
2. Accuracy_Epoch_Plot_inCV.m	- Compute prediction performances for all the checkpoint networks (which can be used for plotting training curve). 
					The checkpoint network that optimizes the validation accuracy is use for performance evaluation. 
3. ComputingGradients_inCV.m - Take the optimal checkpoint network and evaluate the average gradient attribution map.

4. GraphCNN.m and GraphCNNwithCovariates.m - Both are scripts for constructing the BrainNetCNN, where GraphCNNwithCovariates.m is the version with additional regressors

And the .mat files generated are:
1. DLResultsForAllPredictionTasks.mat - A struct variable with all the prediction results
2. DLGradAndNetForAllPredictionTasks.mat - A struct variable with all the gradient maps and optimal networks, use the ConnectomeToDataMat function to extract the non-zero entries,
							the resultant array will have the same shape as the beta maps from classical ML methods

Run ML_AllPredictionTasks.m to get all the classical ML results


And the .mat files generated are:
1. MLResultsForAllPredictionTasks.mat - A struct variable with all the prediction results
2. MLBetasForAllPredictionTasks.mat - A struct variable with all the beta maps
