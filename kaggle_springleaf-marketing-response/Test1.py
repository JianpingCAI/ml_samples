import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplt
import gc


train = pd.read_csv('./springleaf-marketing-response/train_small.csv')
train.head()
train.drop(['ID'], axis=1, inplace=True)

#smaller set
train.drop(train.columns[50:-1], axis=1, inplace=True)
train.drop(train.index[1000:], inplace=True)

train_y = train.pop('target')


# TODO: categorical columns
# 1) data missing rate of each numerical column
desc = train.describe().T
desc['missing %'] = 1-(desc['count']/len(train))
desc.head()

desc = desc[desc['missing %'] < 0.5]
train = train[desc.index]

# 2) variation
desc = train.describe().T
desc = desc[desc['std'] > 0.01]  # 0.01
train = train[desc.index]


# 3) pairwise correlation
# corrMatrix = train.corr() #very slow
corrMatrix = pd.DataFrame(np.corrcoef(
    train.values, rowvar=False), columns=train.columns)
obj = sn.heatmap(corrMatrix, annot=False)
figure = obj.get_figure()
figure.savefig("corrMatrix.png")

del obj
del figure

corrTol = 0.5
for col in corrMatrix:
    if col in corrMatrix.keys():
        thisCol = []
        thisVars = []

        for i in range(len(corrMatrix)):
            if abs(corrMatrix[col][i]) == 1.0 and col != corrMatrix.keys()[i]:
                thisCorr = 0
            else:
                pass
                # thisCorr = (1 if abs(
                #     corrMatrix[col][i]) > corrTol else -1) * abs(temp[corrMatrix.keys()[i]])
            thisCol.append(thisCorr)
            thisVars.append(corrMatrix.keys()[i])

        mask = np.ones(len(thisCol), dtype=bool)
        ctDelCol = 0

        for n, j in enumerate(thisCol):
            mask[n] = not(j != max(thisCol) and j >= 0)

            if j != max(thisCol) and j >= 0:
                corrMatrix.pop('%s' % thisVars[n])
                #temp.pop('%s' % thisVars[n])
                ctDelCol += 1

        corrMatrix = corrMatrix[mask]
