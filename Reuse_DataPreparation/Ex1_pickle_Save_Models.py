"""
"""
# example of creating a test dataset and splitting it into train and test sets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# summarize the scale of each input variable
for i in range(X_test.shape[1]):
	print('>%d, train: min=%.3f, max=%.3f, test: min=%.3f, max=%.3f' %
		(i, X_train[:, i].min(), X_train[:, i].max(),
			X_test[:, i].min(), X_test[:, i].max()))

"""
2. Scale the Dataset
"""
# example of scaling the dataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# define scaler
scaler = MinMaxScaler()
# fit scaler on the training dataset
scaler.fit(X_train)

# transform both datasets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# summarize the scale of each input variable
for i in range(X_test.shape[1]):
	print('>%d, train: min=%.3f, max=%.3f, test: min=%.3f, max=%.3f' %
		(i, X_train_scaled[:, i].min(), X_train_scaled[:, i].max(),
			X_test_scaled[:, i].min(), X_test_scaled[:, i].max()))

"""
3. Save Model and Data Scaler
"""
# example of fitting a model on the scaled dataset
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from pickle import dump

# prepare dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# split data into train and test sets
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=1)

# define scaler
scaler = MinMaxScaler()
# fit scaler on the training dataset
scaler.fit(X_train)

# transform the training dataset
X_train_scaled = scaler.transform(X_train)

# define model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train_scaled, y_train)

# save the model
dump(model, open('model.pkl', 'wb'))

# save the scaler
dump(scaler, open('scaler.pkl', 'wb'))

