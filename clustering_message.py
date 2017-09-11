import pandas as pd
import re
import numpy as np
from time import time

from function_definitions.ML.data_preparation.filtering_sms_script import filtered_df_func
from function_definitions.ML.data_analysis.classifier_execution_script import cluster_data
#from function_definitions.ML.data_analysis.classifier_execution_script import predict_test_data
from sklearn.feature_extraction.text import CountVectorizer

def clustering_algo():
	
	print 'in'
	df=pd.read_csv('data_files/bank_sms_classified.csv')
	print 'out'
	
	"""
	cluster_data function is the main function that gets called that has all the functions. First 2000 rows are used for testing. 
	"""
	cluster_data(df.head(2000))
	
	
	#text_classify(msg_dict,'MessageType',train_df)
	
	
	
	#text_classify(train_df,test_df)
	
	
	

clustering_algo()