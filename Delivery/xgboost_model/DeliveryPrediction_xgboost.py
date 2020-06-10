import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from numpy import sort
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot

# load data files
train_dataset = pd.read_excel('../data_train.xlsx')
test_dataset = pd.read_excel('../data_test.xlsx')

train_dataset.shape
train_dataset.describe()
train_dataset.head()

# data processing
target_column = 'Delivery Time (days)'
# drop some unuseful columns
drop_columns = ['Order ID', 'Product Center', 'Customer Expected Delivery Date',
                'Customer Name', 'Order Create Date']


def preprocess_data(df_dataset):
    '''
    data preprocess
    '''
    #data = pd.get_dummies(dataset, dummy_na=True, drop_first=True)
    # remove non-useful columns
    df_dataset = df_dataset.drop(drop_columns, axis=1)

    X = df_dataset.drop(target_column, axis=1)
    y = df_dataset[target_column]

    # for X: deal with missing categorical values
    categorical_feature_mask = X.dtypes == object
    categorical_cols = X.columns[categorical_feature_mask].tolist()
    X[categorical_cols] = X[categorical_cols].apply(
        lambda col: col.fillna("NA"))

    # for y: fill missing values with mean
    y.fillna(round(y.mean()), inplace=True)

    return X, y


def encode_categorical_columns(df_train, df_test):
    '''
    label encode categorical columns
    '''
    encoders = {}
    df_combined = df_train.append(df_test)
    categorical_feature_mask = df_combined.dtypes == object
    categorical_cols = df_combined.columns[categorical_feature_mask].tolist()

    for col in categorical_cols:
        encoders[col] = LabelEncoder().fit(df_combined[col])

    for col in categorical_cols:
        df_train[col] = encoders[col].transform(df_train[col])
        df_test[col] = encoders[col].transform(df_test[col])

    return encoders


# train & test data
X_train, y_train = preprocess_data(train_dataset)
X_test, y_test = preprocess_data(test_dataset)
encoders = encode_categorical_columns(X_train, X_test)

# xgboost model
model = XGBRegressor(max_depth=3)
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

# ML training
model.fit(X_train, y_train)

# plot feature importances
print(model.feature_importances_)
plot_importance(model)
pyplot.show()
# pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
# pyplot.show()

# inference
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# error staticstics
mse = mean_squared_error(y_test, predictions, squared=True)
print("test mse: %.2f" % (mse))

diffs = predictions-y_test.values
num_underestimate = np.sum(diffs < 0)
num_overestimate = np.sum(diffs > 0)
num_correct = np.sum(diffs == 0)

print("under-estimated:   {}, {:.2f}%".format(num_underestimate,
                                              num_underestimate/len(diffs)*100))
print("over-estimated :   {}, {:.2f}%".format(num_overestimate,
                                              num_overestimate/len(diffs)*100))
print("correct-estimated: {}, {:.2f}%".format(num_correct, num_correct/len(diffs)*100))


# thresholds = sort(model.feature_importances_)
# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(X_train)
#     # train model
#     selection_model = XGBRegressor(max_depth=3)
#     selection_model.fit(select_X_train, y_train)
#     # eval model
#     select_X_test = selection.transform(X_test)
#     y_pred = selection_model.predict(select_X_test)
#     predictions = [round(value) for value in y_pred]

#     mse = mean_squared_error(y_test, predictions)
#     print("Thresh=%.3f, n=%d, mse: %.2f%%" %
#           (thresh, select_X_train.shape[1], mse))
