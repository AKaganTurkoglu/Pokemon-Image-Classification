
from mlxtend.plotting import plot_decision_regions
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import h5py


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

#feature-descriptor-3: color histogram
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 254, 0, 254, 0, 254])
    cv2.normalize(hist, hist)
    return hist.flatten()

fixed_size       = tuple((120, 120))
train_path       = "backup"
h5_data          = 'output/data.h5'
h5_labels        = 'output/labels.h5'
bins             = 16

train_labels = os.listdir(train_path)
train_labels.sort()

global_features = []
labels          = []

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    path = "C:/Users/Yusuf/Desktop/Uning/4-1/veri madenciligi/proje/backup/" + training_name + "/"

    # loop over the images in each sub-folder
    for filename in os.listdir(path):
        # get the image file name
        file = path + filename

        # read the image
        image = cv2.imread(file)

        fv_histogram  = fd_histogram(image)
        fv_haralick = fd_haralick(image)
        fv_hu_moments = fd_hu_moments(image)
        # global_feature = fv_histogram
        
        global_feature = np.hstack([fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))


print("[STATUS] completed Global Feature Extraction...")


# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# scale features in the range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1:', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1:', data=np.array(target))



h5f_data.close()
h5f_label.close()


print("[STATUS] end of training..")


