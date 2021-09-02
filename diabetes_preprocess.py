from diabetes_util import *

def preprocess_data(data):
	print('Shape Before Process: ' + str(data.shape))
	##########################################################################

	## The process for the outlier rejection (P)

	data = diabetes_util.outlier_Rejection (data,
		          iqr_Mean=False,
		          iqr_Medain=False,
		          iqr=True,
		          manual=False)
	print('Shape After outlier Removed: ' + str(data.shape))

	##########################################################################

	## The process for the filling missing values (Q)


	'''
	A fasting insulin level should never be 0, which it might be in a person with untreated Type 1. 
	It shouldn't go below 3. 
	But a high insulin level is just as problematic. 
	A high insulin level is a sign of insulin resistance or prediabetes.
	'''

	for col in ['F2', 'F3', 'F4', 'F5', 'F6']:   
	    diabetes_util.replace_zero(data, col, 'Outcome')    #replaced by mean according to each class
	print('Shape After Filling Missing Value: ' + str(data.shape))
	     

	##########################################################################
	#  algo parameters are
	# 'PCA','ICA','corr','None'

	X_Data,Y_Lavel = diabetes_util.feature_Selector(data, algo='corr', n_feature=6)    
	print('Shape After Feature Selection: ' + str(data.shape))

	##########################################################################
	# The process of Standardization  (S)
	# scaler =  preprocessing.StandardScaler()
	# X_Data,Y_Lavel= scaler.fit_transform(X_Data), Y_Lavel
	# print('Shape After Standardization: ' + str(data.shape))          


	##########################################################################
	# Stratified K-Folds cross-validator
	# Provides train/test indices to split
	# data in train/test sets.This cross-validation
	#  object is a variation of KFold that returns 
	#  stratified folds. The folds are made by preserving 
	#  the percentage of samples for each class.

	kf = StratifiedKFold(n_splits=5,
		             shuffle=False,
		             random_state=random_initializer)
		             
	data_plot (data,
		Pair_plot=True,
		Dist_Plot=True,
		Plot_violinplot=True,
		Plot_confusionMatrix=True,
		box_Gaussian=False)
