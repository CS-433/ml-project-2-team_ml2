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
    The preprocess of the train and test data set is done by running the file preprocess.py before running the file run.py.
    It does multiple tasks :
        3.1 : Data augmentation of the train dataset :
            - Add 90/180/270° rotation of each original images  (x4 the number of images)
            - Add 45° rotation of each resulting images (x2 the number of images)
            - Add 180° flip of each images (x2 the number of images)
            Therefore it multiply by 16 the number of images in the train dataset.
        3.2 : Normalization (Z-Score) :
            - The train and test dataset are normalized via a Z-score method
                Normalized train image = [ Original train image - (mean train image) ] / standard deviation train image
                Normalized test image = [ Original test image - (mean test image) ] / standard deviation test image

4. Submission

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
			Net.py ??????????????????????????????
			ResUNet.py
			UNet.py
			UNet_parts.py
			UNet_old.py ?????????????????????????
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
	run.py
	final_submission.csv ???????????????????????
	
