import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import os
import glob
import numpy as np

def get_mean_median(image_folder, image_format):
    """
    Dataset mean and median histogram
    """
    hist_set = np.empty((0, 256), int)

    for image in glob.glob(os.path.join(image_folder, image_format)):
        im_array = np.asarray(Image.open(image))
        hist, _ = np.histogram(im_array, bins=256)
        hist_set = np.append(hist_set, hist.reshape((1, 256)), axis=0)

    mean = np.mean(hist_set, axis=0)
    median = np.median(hist_set, axis=0)
    return mean, median


def savefig_correlation_matrix(image_folder, image_format):
    """
    Export dataset correlation coefficient matrix as image
    *not sure if get the question correctly.
    """
    # data matrx
    X = np.empty((0, 256), int)

    for image in glob.glob(os.path.join(image_folder, image_format)):
        im_array = np.asarray(Image.open(image))
        hist, _ = np.histogram(im_array, bins=256)
        X = np.append(X, hist.reshape((1, 256)), axis=0)

    df = pd.DataFrame(X)
    plt.matshow(df.corr())
    # plt.show()
    plt.savefig('corr.png')
    
# hist_mean, hist_median
#image_folder = 'image_folder_path'
hist_mean, hist_median = get_mean_median(image_folder, '*.png')

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(hist_mean, label='mean')
ax.plot(hist_median, label='median')
ax.legend()
plt.title('Internal-Dataset-PNG')
plt.show()

# save corr matrix
savefig_correlation_matrix(image_folder, '*.png')