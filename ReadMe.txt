===== AUTHORS =====
by Laurent Gürtler, Anthony Ducret and Swann Destouches

===== FUNCTIONNING =====
1. The training, evaluation and prediction randering is done from the file run.py at the root
	1.1 In the section <TRAINING NEURAL NETWORK>, the different hyperparameters can be chosen:
		- image_size
		- model_type (??????)
		- number of epochs
		- fraction of data used
		- device used (cuda or cpu)
	1.2 In the section <LOAD MODEL + PREDICTION>, a flag can be set to 'True' or 'False' in order to do predictions or not.
2. Multiple models can be run:
	2.1 For the UNet or the ResUNet, simply precise the model name in the run.py file
	2.2 For the CUNet, the flag 'useCUNet' must be set to 'True' at the beginning of the file. the model choice in the hyperparameters will not be taken into account
3. Preprocess
	3.1 
4. Submission
	4.1 the file 'get_prediction.py' from the folder 'src/Submission' apply the model on the test set and create the submission file for AICrowd
	4.2 The model is saved in the directory 'Predictions' and the output of the model on the tests set are saved in the sub-directory 'Predictions/images'
	4.3 The csv file containing the submission to AICrowd is saved in the directory 'Submission' under the name 'final_submission.csv'

===== ARCHITECTURE =====
./
	Data/
		test_set_images/
		training/
		training_processed/
	Predictions/
		images/
		model.pth
	Results/
		Prediction_imgs/
			CUNet/
				Final_Results/
				inter_training_dataset/
				pred_MUNet/
				pred_UNet/
				temp/
	src/
		Modeles/
			MDUNet.py
			ResUNet.py
			UNet.py
			UNet_parts.py
		Preprocess/
			preprocess.py
		Save_Load/
			load_data.py
			save_data.py
		Submission/
			mask_to_submission.py
			submission_to_mask.py
		training.py
		trianing_CUNet.py
	Submissions/	
		final_submission.csv
	run.py

	
