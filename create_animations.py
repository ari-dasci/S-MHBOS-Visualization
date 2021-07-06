import sys
from mhbos import MHBOS
from scipy.io import loadmat
import os
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix as confusion_matrix_function
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import time
import matplotlib.pyplot as plt


# Fix the seed
random.seed(123456789)
np.random.seed(123456789)

ALGORITHM_NAME = "MHBOS"

################################################################################
##                          AUXILIARY FUNCTIONS                               ##
################################################################################

def readDataset(route):
    '''
    @brief This function reads the datasets in matlab format and gets the data and labels
    @param route Route to the dataset
    @return It returns first the dataset and then the labels
    '''
    mat = loadmat(route)
    sc = StandardScaler(copy=False)
    return sc.fit_transform(mat["X"].astype("float64")), mat["y"].flatten().astype("int64")

def sliceDataset(X,nslices):
    '''
    @brief This function slices the X and y in nslices slices
    @param X Data
    @param nslices Number of subdatasets we want to make
    @return This function returns two lists of slices of the original dataset. The first
    one are the data slices and the second one are the labels slices.
    '''
    X_slices = []
    slice_size=len(X)//nslices
    cont = 0
    for i in range(nslices-1):
        cont+=slice_size
        X_slices.append(X[:cont])
    X_slices.append(X[cont:])
    return X_slices

################################################################################
##                               COMPARISON                                   ##
################################################################################

precisions, recalls, f1s, confusion_matrices, aucs, fprs, tprs = [], [], [], [], [], [], []
route = "shuttle.mat"
parameters = [{'nbins': 25, 'epsilon': "auto", 'alpha': "auto", 'histogram_combination': 'minmax', 'nslices': 10, 'behaviour': 'static'},
            {'nbins': 25, 'epsilon': "auto", 'alpha': "auto", 'histogram_combination': 'fixedwidth', 'nslices': 10, 'behaviour': 'static'},
            {'nbins': 25, 'epsilon': "auto", 'alpha': "auto", 'histogram_combination': 'fuzzycombination', 'nslices': 10, 'behaviour': 'static'},
            {'nbins': 25, 'epsilon': "auto", 'alpha': "auto", 'histogram_combination': 'adjustinglimits', 'nslices': 10, 'behaviour': 'dynamic'},
            {'nbins': 25, 'epsilon': "auto", 'alpha': "auto", 'histogram_combination': 'histogramfusion', 'nslices': 10, 'behaviour': 'dynamic'}]

# For each dataset
for ind, param in enumerate(parameters):
    # Read it
    X,y = readDataset("./datasets/" + route)
    num_anomalies = int(np.sum(y))
    anomalies_perc = num_anomalies/len(y)
    min = np.amin(X, axis=1)
    max = np.amax(X, axis=1)

    model = MHBOS(nbins=int(param["nbins"]), epsilon=param["epsilon"], alpha=param["alpha"], behaviour=param["behaviour"],
            density=None, contamination=anomalies_perc, histogram_combination=param["histogram_combination"])
    X_slices = sliceDataset(X,param["nslices"])
    cont = 0
    for Xs in X_slices:
        model.fit(Xs,min=min,max=max)
        bins, counts = model.bins_edges[0], model.histograms[0]
        counts = counts/np.amax(counts)
        left_edges = bins[:-1]
        right_edges = bins[1:]
        widths = (right_edges-left_edges)
        widths = widths*0.95
        plt.figure(1)
        plt.bar(left_edges,counts,width=widths,align="edge",color="blue")
        plt.xlabel('Bins')
        plt.ylabel('Counts')
        plt.title("Histogram")
        plt.legend(loc='best')
        plt.savefig(param["histogram_combination"]+"_"+str(cont)+".png")
        plt.close(1)
        cont+=1
