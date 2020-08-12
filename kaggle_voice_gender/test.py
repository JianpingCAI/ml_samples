# libraries
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
# from feature_selector import FeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
sns.set()  # Making seaborn the default styling

df = pd.read_csv('voice.csv')
df.head()

label_encoder = LabelEncoder()
df.label = label_encoder.fit_transform(df.label)

df.info()
df.describe()

X = df.drop('label', axis=1)
y = df.label

# testing data size at 20%
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=4)
x_train.shape

# filter-0. persent missing values


# filter-1. zero variance (unique values)
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(x_train)
print(x_train.columns[constant_filter.get_support()])
x_train = x_train[x_train.columns[constant_filter.get_support()]]
x_train.shape
x_train.var()

# Correlation matrix for all independent vars
corrMatrix = x_train.corr()
# #faster
# corrMatrix2 = pd.DataFrame(np.corrcoef(
#     x_train.values, rowvar=False), columns=x_train.columns)


# filter-2. pairwise correlation
# threshold setting
corrTol = 0.75

absCorrWithDep = []
for var in corrMatrix.keys():
    absCorrWithDep.append(abs(y.corr(x_train[var])))

ds_absCorrWithDep = pd.Series(absCorrWithDep, index=corrMatrix.keys())

# for each column in the corr matrix
orignColNames = corrMatrix.keys()
for col_name in orignColNames:

    if col_name in corrMatrix.keys():

        del_names = set()

        # Store the corr with the dep var for fields that are highly correlated with each other
        for iRow in range(len(corrMatrix)):

            row_name = corrMatrix.keys()[iRow]
            abs_value = abs(corrMatrix[col_name][iRow])

            if (col_name != row_name) and (abs_value >= corrTol):

                del_name = (row_name if ds_absCorrWithDep[row_name] <= ds_absCorrWithDep[col_name]
                            else col_name)
                del_names.add(del_name)

                if(col_name == del_name):
                    del_names = {col_name}
                    break

        if len(del_names) > 0:
            list_del_names = list(del_names)
            corrMatrix = corrMatrix.drop(del_names, axis=1)
            corrMatrix = corrMatrix.drop(del_names)

print(corrMatrix.columns)
x_train = x_train[corrMatrix.columns]

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 14))
ret_sns = sns.heatmap(x_train.corr(), cmap='Reds',
                      annot=True, linewidths=.5, ax=axes[0], yticklabels=True, xticklabels=True)
figure = ret_sns.get_figure()

# filter-3. Multicollinearity
# ref: https://xavierbourretsicotte.github.io/stats_inference_2.html
# Condition Index : the square root of the ratio between
# the largest eigenvalue and the eigenvalue associated to each variable.
# The condition index is equal to 1 if variables are perfectly independent,
# a large value > 30 highlights a collinearity issue.
ct = len(corrMatrix)
minVarsKeep = 10
condIndexTol = 30
if ct > minVarsKeep:
    while True:
        ct -= 1
        col_names = corrMatrix.keys()

        # condition index
        eigVals, eigVecs = np.linalg.eig(corrMatrix)
        condIndex = (max(eigVals)/eigVals)**0.5

        if max(condIndex) <= condIndexTol or ct == minVarsKeep:
            break

        # minEigVal = min(eigVals)
        idx_minEigVal = np.argmin(eigVals)  # min value, close to zero
        # minEigVal = eigVals[idx_minEigVal]

        absEigVec = abs(eigVecs[:, idx_minEigVal])
        idx_maxAbsEigVec = np.argmax(absEigVec)
        # maxAbsEigVec = absEigVec[idx_maxAbsEigVec]

        mask = np.ones(len(corrMatrix), dtype=bool)
        mask[idx_maxAbsEigVec] = False
        # for iM, col in enumerate(corrMatrix.keys()):
        #     mask[iM] = (iM != idx_maxAbsEigVec)

        # delete row
        corrMatrix = corrMatrix[mask]
        # delete column
        corrMatrix.pop(col_names[idx_maxAbsEigVec])


# filter-4. PCA

        # filter-5.

        # fs = FeatureSelector(data=x_train, labels=y_train)
        # fs.identify_zero_importance(task='classification', eval_metric='auc')

        # RandomForestClassifier
model_RF = RandomForestClassifier(random_state=42)
model_RF.fit(x_train, y_train)

features = x_train.columns
importances = model_RF.feature_importances_
indices = np.argsort(importances)

y_range = range(len(indices))
plt.title('Feature Importances')
plt.barh(y_range, importances[indices], color='b', align='center')
plt.yticks(y_range, [features[i] for i in indices])
plt.xlabel('Relative Importance')
# plt.show()

fig.savefig("corrMatrix.png")


# XGBoost
model_XGB = XGBClassifier(max_depth=3, seed=42)
# model = XGBRegressor(colsample_bytree=0.4,
#                      gamma=0,
#                      learning_rate=0.07,
#                      max_depth=5,
#                      min_child_weight=1.5,
#                      n_estimators=5000,
#                      reg_alpha=0.75,
#                      reg_lambda=0.45,
#                      subsample=0.6,
#                      seed=42)
model_XGB.fit(x_train, y_train)

plot_importance(model_XGB)
plt.show()

# LogisticRegression
model_logr = LogisticRegression()
model_logr.fit(x_train, y_train)
x_test = x_test[x_train.columns]
pred_y = model_logr.predict(x_test)
accuracy = metrics.accuracy_score(pred_y, y_test)
print("LR: %.3f" % accuracy)

# RandomForest
# model_RF = RandomForestClassifier()
model_RF.fit(x_train, y_train)
x_test = x_test[x_train.columns]
pred_y = model_RF.predict(x_test)
accuracy = metrics.accuracy_score(pred_y, y_test)
print("RF: %.3f" % accuracy)

# XGBoost
# model_XGB = XGBClassifier(max_depth=3)
# model_XGB.fit(x_train, y_train)
x_test = x_test[x_train.columns]
pred_y = model_XGB.predict(x_test)
accuracy = metrics.accuracy_score(pred_y, y_test)
print("XGB: %.3f" % accuracy)

# Gaussian Naive Bayes
model_GNB = GaussianNB()
model_GNB.fit(x_train, y_train)
x_test = x_test[x_train.columns]
pred_y = model_GNB.predict(x_test)
accuracy = metrics.accuracy_score(pred_y, y_test)
print("GNB: %.3f" % accuracy)

model_SVC = SVC()
model_SVC.fit(x_train, y_train)
x_test = x_test[x_train.columns]
pred_y = model_SVC.predict(x_test)
accuracy = metrics.accuracy_score(pred_y, y_test)
print("SVC: %.3f" % accuracy)

# import statsmodels.api as statsmodel
# model_OLS = statsmodel.OLS(y_train, x_train)
# results = model_OLS.fit()
# results.summary()
# x_test = x_test[x_train.columns]
# pred_y = model_OLS.predict(x_test)