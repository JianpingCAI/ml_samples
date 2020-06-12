from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
sns.set()

wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Categorical.from_codes(wine.target, wine.target_names)

X.head()
wine.target_names
wine.target

df = X.join(pd.Series(y, name='class'))


LinearDiscriminantAnalysis?

lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)


lda.explained_variance_ratio_

le = LabelEncoder()
y = le.fit_transform(df['class'])

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_lda[:, 0],
    X_lda[:, 1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)


