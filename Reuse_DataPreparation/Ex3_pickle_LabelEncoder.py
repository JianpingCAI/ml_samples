"""
save encoder, using pickle
"""
# libraries
# from xgboost import plot_importance
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
import pandas as pd
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_selection import VarianceThreshold
# # from feature_selector import FeatureSelector
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# sns.set()  # Making seaborn the default styling

df = pd.read_csv('../kaggle_voice_gender/voice.csv')
df.head()

label_encoder = LabelEncoder()
df['label2'] = label_encoder.fit_transform(df.label)

pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))


import pickle
df = pd.read_csv('../kaggle_voice_gender/voice.csv')
df.head()

loaded_lable_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

df['label2']  = loaded_lable_encoder.fit_transform(df.label)
df.head()
