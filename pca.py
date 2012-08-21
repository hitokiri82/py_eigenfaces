""" Principal Component Analysis.
INCLUDE SOME COMMENT HERE ABOUT WHY THIS
IMPLEMENTATION WORKS FOR BIG DATA SETS

(c) Copyright 2011, Francisco Caraballo La Riva.
This program is free software: you can redistribute it and/or modify
it in any way.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
"""

from copy import deepcopy
import numpy as np

def pca(input,normalise=1,threshold = 0):

    data = deepcopy(input)

    m = np.mean(data,axis=0)
    data -= m

    data = data.T

    C = np.dot(data.T,data)

    evals,evecs = np.linalg.eig(C)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]
    evecs = np.dot(data,evecs)

    if normalise:
        for i in range(np.shape(evecs)[1]):
            evecs[:,i] / np.linalg.norm(evecs[:,i]) * np.sqrt(evals[i])

    evecs = evecs.T

    if threshold:
        total = np.sum(evals)
        s = 0
        i = 0
        while (s/total) < threshold:
                s += evals[i]
                i += 1
        evecs = evecs[:i,:]
        evals = evals[:i]

    return evecs, evals