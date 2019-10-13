import numpy as np
import os
import pandas as pd
import cv2 as cv

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

cs = ['white','red','yellow','orange','blue','green']
training_data_folder = './training_data/'
pickled_model = './clf.joblib'

def reshape_images(training_images, faces_per_image = 6):
    # Number of images and squares inside of them
    n_img = len(training_images)
    n_squares = 9 * faces_per_image * n_img
    faces = np.zeros((faces_per_image * n_img, 150, 150, 3), dtype='uint8')

    for j, d in enumerate(training_images):
        for i in range(faces_per_image):    
            faces[n_img * i + j] = d[i,:,:,:]

    # generate training data
    all_squares = [square for img in faces for clmn in np.hsplit(np.flipud(img), 3) for square in np.vsplit(clmn,3)]
    all_square_means_rgb = [np.mean(square, axis = (0,1)) for square in all_squares]
    all_square_means_rgb = np.array(all_square_means_rgb).astype('uint8')
    all_square_means_yuv = cv.cvtColor(all_square_means_rgb.reshape(1, n_squares, 3), cv.COLOR_RGB2YUV)
    all_square_means_yuv = np.squeeze(all_square_means_yuv)
    all_square_means_uv  = all_square_means_yuv[:,1:3]

    X_rgb = all_square_means_rgb.astype('float')
    X_yuv = all_square_means_yuv.astype('float')
    
    return X_yuv 

def get_training_images():    
    training_data_files = [f for f in os.listdir(training_data_folder) 
                           if 'training_data' in f]
    training_images = []
    for f in training_data_files:
        d = np.load("{0}/{1}".format(training_data_folder, f))    
        training_images.append(d)

    return training_images
        
def get_training_data():
    training_images = get_training_images()
    
    #
    n_img = len(training_images)
    n_squares = 9 * 6 * n_img
    
    # Color labels
    clrs = [k * np.ones(int(n_squares / 6), dtype='int') for k in range(6)]
    clrs = np.concatenate(clrs)
    clrs_labels = [cs[k] for k in clrs]

    X_yuv = reshape_images(training_images)

    return X_yuv, clrs

def get_classifier():
    # if model is not trained, train and pickle for future use
    if not os.path.exists(pickled_model):        
        X_yuv, clrs = get_training_data()    
        clf_yuv = svm.SVC(gamma='scale', kernel='poly', degree=5, probability=True)
        clf_yuv.fit(X_yuv, clrs)
        dump(clf_yuv, pickled_model)

    # load pickled model
    else:
        clf_yuv = load(pickled_model)

    return clf_yuv

def label_images(clf_yuv, images, faces_per_image = 6):
    # convert data into correct format
    X_yuv = reshape_images(images, faces_per_image)

    # predict
    pred_yuv  = clf_yuv.predict(X_yuv).astype('uint32')    
    class_proba_yuv = clf_yuv.predict_proba(X_yuv)
    pred_colors_yuv = [cs[k] for k in pred_yuv]
    pred_proba_yuv  = [class_proba_yuv[i,k] for i,k in enumerate(pred_yuv)]

    return pred_colors_yuv, pred_proba_yuv


if __name__ == '__main__':
    #clf_yuv = get_classifier()
    #X_yuv, clrs = get_training_data()
    training_images = get_training_images()
    print(training_images[0].shape)
    #pred_colors_yuv, pred_proba_yuv = label_images(clf_yuv, training_images)
    #print(pred_colors_yuv)
