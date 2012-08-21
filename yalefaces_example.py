"""
(c) Copyright 2011, Francisco Caraballo La Riva.
This program is free software: you can redistribute it and/or modify
it in any way.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
"""

from scipy.misc import imread
import numpy as np
from copy import deepcopy
import pca
import classifiers
from glob import glob

def load_images_bw(filenames):
        result = []
        for i in filenames:
                result.append(imread(i, flatten = 1))
        return result
# This path must point to the location of the pyfaces files in your system.
# Watch out for a file called saveddata.cache that might be in that folder an is not an image        
image_vectors = load_images_bw(glob("C:\Users\Pakiko\Documents\UPM\Robots Aut\pyfaces\yalefaces\yalefaces\*"))

# Images are represented as a vector of all the pixels.
length = image_vectors[0].shape[0]*image_vectors[0].shape[1]
for v in image_vectors:
        v.shape = (1,length)

vt = tuple(image_vectors)
vectors = np.vstack(vt)
# Do Principal Component Analysis on the matrix of the 
# images
eig_vectors,eig_vals = pca.pca(vectors)
transformed_vectors = (np.dot(eig_vectors,vectors.T)).T

# This is just a hack to get the tags vector
# I just happen to know that the tag changes every
# 11 images because of how I loaded the images.
tags = [ ]
for i in range(transformed_vectors.shape[0]):
        tags.append(i/11)
tags = np.asarray(tags)
tags.shape = (tags.shape[0],1)

K_NEIGHBORS = 3
#Leave one out test of the classifier
successes = 0.0
for i in range(transformed_vectors.shape[0]):
    vec = np.vstack((transformed_vectors[:i,:],transformed_vectors[i+1:,:]))
    tags_2 = np.vstack((tags[:i,:],tags[i+1:,:]))
    classifier = classifiers.knnclassifier(vec,tags_2,K_NEIGHBORS)
    classifier_output = classifier.classify(transformed_vectors[i,:])
    if classifier_output == tags[i,0]:
        successes += 1
    #print 'Classifier output: ', classifier_output, 'Real Tag: ', tags[i,0]
print 'Success %', successes/transformed_vectors.shape[0]*100

