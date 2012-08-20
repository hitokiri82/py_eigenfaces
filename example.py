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
# Each set contains 18 images tagged with 
# the same label
images1 = load_images_bw(glob("set1/*"))
images2 = load_images_bw(glob("set2/*"))
images3 = load_images_bw(glob("set3/*"))

image_vectors = images1 + images2 + images3

# Images are represented as a vector of all the pixels.
# In this example, images were 400 x 400 so the resulting 
# vectors are 1 x 160000
for v in image_vectors:
        v.shape = (1,160000)

vt = tuple(image_vectors)
vectors = np.vstack(vt)
# Do Principal Component Analysis on the matrix of the 
# images
eig_vectors,eig_vals = pca.pca(vectors, threshold=0.99)
transformed_vectors = (np.dot(eig_vectors,vectors.T)).T

# This is just a hack to get the tags vector
# I just happen to know that the tag changes every
# 18 images because of how I loaded the images.
tags = [ ]
for i in range(transformed_vectors.shape[0]):
        tags.append(i/18)
tags = np.asarray(tags)
tags.shape = (54,1)

#Leave one out test of the classifier
successes = 0.0
for i in range(transformed_vectors.shape[0]):
    vec = np.vstack((transformed_vectors[:i,:],transformed_vectors[i+1:,:]))
    tags_2 = np.vstack((tags[:i,:],tags[i+1:,:]))
    classifier = classifiers.knnclassifier(vec,tags_2,5)
    classifier_output = classifier.classify(transformed_vectors[i,:])
    if classifier_output == tags[i,0]:
        successes += 1
    print 'Classifier output: ', classifier_output, 'Real Tag: ', tags[i,0]
print 'Success %', successes/transformed_vectors.shape[0]*100
