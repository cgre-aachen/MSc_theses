'''Algorithm to cluster 1D-well log data

# criteria for boundaries: maxima and minima (derivative = 0)
# criteria for grouping --> minimal variance
# name layers with means to compare with boreholes

'''

import numpy as np
import pandas as pb
import itertools as it


class Clustering:
    def __init__(self, data, features, borehole_name):

        self.data = data
        self.features = features

        '''self.data_ex = data.extract_data(self.data,self.features,borehole_name)'''
        self.data.normalize_feature_vectors()
        self.features = features

    def extract_data(self,data,features,borehole_name):
        return  data.loc[np.where(data == borehole_name)[0], features]  # extract all feature of one borehole

    def normalize_feature_vectors(feature_vectors):  # function to normalize data (x - mean / sigma)
        return (feature_vectors - np.mean(feature_vectors, axis=0).T) / np.std(feature_vectors, axis=0)

def find_boundaries(self):
    self.diff = self.data[:-2]-self.data[1:]



def test_bic(feature_vectors_norm):
    # Bayesian information criteria
    nft = []
    for k in range(2, 25):
        nft.append(bic(feature_vectors_norm, k,
                       plot=False))  # Investigate the number of labels (one label can include several cluster)
    nf = max(set(nft), key=nft.count)  # put out the most common value in bic
    print('The optimal number of layers is: ', nf)
    return nf, nft

