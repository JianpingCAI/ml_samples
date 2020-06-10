import torch
import random
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

class MLPRegressor(nn.Module):
    '''
    MLP neural network model
    '''

    def __init__(self, dim_input):
        super().__init__()
        self.fc1 = nn.Linear(dim_input, 8)
        self.fc2 = nn.Linear(8, 20)
        # self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(20, 1)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):

        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        # x = self.dropout(F.relu(self.fc3(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)


# data processing
target_column = 'Delivery Time (days)'
# drop_columns = ['Order ID', 'Product Center', 'Customer Expected Delivery Date',
#                 'Customer Name', 'Order Create Date']
drop_columns = ['Order ID', 'Product Center', 'Customer Expected Delivery Date',
                'Customer Name', 'Order Create Date', 'Product Supplier']


def data_preprocess(df_dataset):
    '''
    Data preprocessing: 
    - drop some unuseful columns 
    - deal with missing values
    '''
    #data = pd.get_dummies(dataset, dummy_na=True, drop_first=True)

    # remove unuseful columns
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


def model_inference(model, test_X, targets_y):
    # inference on test dataset
    test = torch.from_numpy(test_X.values).float()

    with torch.no_grad():
        model.eval()
        output = model.forward(test)

    # statistics
    predictions = output.numpy()
    predictions = [round(value) for value in predictions.flatten()]
    mse = mean_squared_error(targets_y.values, predictions, squared=True)
    print("test mse: %.2f" % (mse))

    diffs = predictions-targets_y.values
    num_underestimate = np.sum(diffs < 0)
    num_overestimate = np.sum(diffs > 0)
    num_correct = np.sum(diffs == 0)
    print("under-estimated:   {}, {:.2f}%".format(num_underestimate,
                                                    num_underestimate/len(diffs)*100))
    print("over-estimated :   {}, {:.2f}%".format(num_overestimate,
                                                    num_overestimate/len(diffs)*100))
    print("correct-estimated: {}, {:.2f}%".format(num_correct,
                                                    num_correct/len(diffs)*100))


# if __name__ == '__main__':
#     pass
