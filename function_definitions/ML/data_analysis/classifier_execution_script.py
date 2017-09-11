import pandas as pd
import re
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import wordnet
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import load_iris 
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
import pickle	
from time import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from function_definitions.ML.data_preparation.filtering_sms_script import filtered_df_func
from scipy import sparse, io
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt





#account_type_dict={'CASA':1,'Credit_Card':2,'Debit_Card':3,'Loan':4,'Prepaid_Card':5,'Wallet':6,'_NA_':7}


def cluster_data(df):
	
	"""
	the filtered_df_func is used for text pre-processing and filtering out noise related data from the text.
	"""
	df=filtered_df_func(df)
	print df
	
	
	
	msg_list=[]
	label_list=[]
	
	
	
	for idx,row in df.iterrows():
		
		#df.at[idx,column_name+'_id']=column_dict[df.at[idx,column_name]]
		msg_list.append(str(df.at[idx,'filtered_msg']))
		#label_list.append(int(df.at[idx,column_name+'_id']))
	

	#-------------------------for account type
	#for idx,row in train_df.iterrows():
		#print train_df.at[idx,'AccountType'],'pppppppp' 
		
		#train_df.at[idx,'Account_Type_id']=account_type_dict[train_df.at[idx,'AccountType']]
		#msg_list.append(str(train_df.at[idx,'filtered_msg']))
		#label_list.append(int(train_df.at[idx,'Account_Type_id']))
	
	
	
	
	#train_df.to_csv('feature_sample.csv')
	#print (train_df.head())
	#print msg_list
	
	#-------------------------for account type
	#for idx,row in test_df.iterrows():
		#test_df.at[idx,'Account_Type_id']=account_type_dict[test_df.at[idx,'AccountType']]
		#test_label_list.append(int(test_df.at[idx,'Account_Type_id']))
	
	#train_df.to_csv('feature_sample.csv')
	#print (train_df.head())
	#print msg_list
	
	
	"""
	initialization of vectorizer
	"""
	vect=CountVectorizer(ngram_range=(1, 2),stop_words='english')
	#vect=HashingVectorizer(ngram_range=(1, 2))
	#tfidf_transformer = TfidfTransformer()
	#vect = TfidfVectorizer(use_idf='False',ngram_range=(1, 2))
	#vect = TfidfVectorizer(stop_words='english')
	"""
	Method to fit and transform
	"""
	t_start = time()
	#print 'start'
	X_train_dtm=vect.fit_transform(msg_list)
	#print X_train_dtm,type(X_train_dtm),'zzzzzzzzzzzzzzzzzz'
	#save_vectorizer = open("vocabulary/clustering.pickle","wb")
	#pickle.dump(vect, save_vectorizer)
	#save_vectorizer.close()
	
	#tmp=pd.DataFrame(X_train_dtm.toarray(),columns=vect.get_feature_names())
	#tmp.to_csv('tfidf.csv')
	#raw_input()
	
	#X_new_counts = vect.transform(msg_list)
	#X_train_tfidf=tfidf_transformer.fit_transform(X_train_dtm)
	#print X_train_dtm
	#print vect.get_feature_names()
	
	
	#tmp=pd.DataFrame(X_train_dtm.toarray(),columns=vect.get_feature_names())
	#print tmp
	#tmp.to_csv('DTM.csv')
	#raw_input()
	
	#sgd=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)	
	#rfc = RandomForestClassifier(n_estimators=100,n_jobs=1, max_depth=None,min_samples_split=2, random_state=0)
	#sgd=SGDClassifier()
	#nb=MultinomialNB()
	#clf = svm.SVC()
	#gnb = GaussianNB()
	wcss=[]
	for i in range(50):
		print 'k :',i
		km=KMeans(n_clusters=i+1).fit(X_train_dtm)
		wcss.append(km.inertia_)
		
	"""
	for plotting the graph of WCSS vs number of clusters 
	"""
	plt.plot(range(1,51),wcss)
	plt.xlabel('Number of clusters')
	plt.ylabel('WCSS value')
	plt.show()
	
	"""
	From the elbow method diagram, the k optimal k value comes to be 25-27. This value is used to cluster the dataset into equal categories.
	"""
	km=KMeans(n_clusters=26).fit(X_train_dtm)
	clusters=km.labels_.tolist()
	t_end = time()
	print (t_end-t_start)/60
	print 'end'
	print len(clusters),len(df)
	print 'yaaaaaaaaaaaaaa'
	cluster_dict={'clusters':clusters}
	cluster_df=pd.DataFrame(cluster_dict)
	result=pd.concat([df,cluster_df],axis=1)
	result.to_csv('clustered_file_final_1.csv')
	
	
	
	
	print ('out')
	

