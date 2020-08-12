
"""
save encoder, using joblib
Joblib is the replacement of pickle as it is more efficent on objects that 
carry large numpy arrays. 
These functions also accept file-like object instead of filenames.
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import pandas as pd

df = pd.read_csv('../kaggle_voice_gender/voice.csv')
df.head()

label_encoder = LabelEncoder()
df['label2'] = label_encoder.fit_transform(df.label)
df.head()

# Save the model as a pickle in a file 
joblib.dump(label_encoder, 'label_encoder.pkl')   


# Load the model from the file 
df = pd.read_csv('../kaggle_voice_gender/voice.csv')
df.head()

loaded_lable_encoder = joblib.load('label_encoder.pkl')

df['label2']  = loaded_lable_encoder.fit_transform(df.label)
df.head()