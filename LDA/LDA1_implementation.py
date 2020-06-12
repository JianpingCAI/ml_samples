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
df.head()

'''
https://towardsdatascience.com/linear-discriminant-analysis-in-python-76b8b17817c2
Linear Discriminant Analysis can be broken up into the following steps:
1.Compute the within class and between class scatter matrices
2.Compute the eigenvectors and corresponding eigenvalues for the scatter matrices
3.Sort the eigenvalues and select the top k
4.Create a new matrix containing eigenvectors that map to the k eigenvalues
5.Obtain the new features (i.e. LDA components) by taking the dot product of the data and the matrix from step 4
'''

# 1.1 within class scatter matrix
class_feature_means = pd.DataFrame(columns=wine.target_names)

for c, rows in df.groupby('class'):
    class_feature_means[c] = rows.mean()

class_feature_means

within_class_scatter_matrix = np.zeros((13, 13))
for c, rows in df.groupby('class'):
    rows = rows.drop(['class'], axis=1)
    s = np.zeros((13, 13))

    for index, row in rows.iterrows():
        x = row.values.reshape(13, 1)
        mc = class_feature_means[c].values.reshape(13, 1)
        x_mc = x - mc
        s += x_mc.dot(x_mc.T)

    within_class_scatter_matrix += s

# 1.2 between class scatter matrix
feature_means = df.mean()

between_class_scatter_matrix = np.zeros((13, 13))
for c in class_feature_means:
    nc = len(df.loc[df['class'] == c].index)
    mc = class_feature_means[c].values.reshape(13, 1)
    m = feature_means.values.reshape(13, 1)
    mc_m = mc-m
    between_class_scatter_matrix += nc * mc_m * mc_m.T

# 2. eigen of the scatter matrix
eigen_values, eigen_vectors = \
    np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(
        between_class_scatter_matrix))

# 3. sort according to the eigenvalues
pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
         for i in range(len(eigen_values))]

pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

for pair in pairs:
    print(pair[0])

# check how much of the variance is explained by each component
eigen_value_sum = sum(eigen_values)

print('Explained Variance')
for i, pair in enumerate(pairs):
    print('Eigenvector {}: {}'.format(i, (pair[0]/eigen_value_sum).real))

# 4. create the matrix
w_matrix = np.hstack((pairs[0][1].reshape(13, 1),
                      pairs[1][1].reshape(13, 1))).real

# project by the matrix
X_lda = np.array(X.dot(w_matrix))

# 5. plot
le = LabelEncoder()
y = le.fit_transform(df['class'])

plt.xlabel('LD1')
plt.ylabel('LD2')

plt.scatter(
    X_lda[:, 0], X_lda[:, 1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)
