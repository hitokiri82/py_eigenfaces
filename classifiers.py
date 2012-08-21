""" Implementation of several classifiers.
(c) Copyright 2011, Francisco Caraballo La Riva.
This program is free software: you can redistribute it and/or modify
it in any way.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
"""

import numpy as np
from scipy import unique

class knnclassifier:
        def __init__(self, ts, tags, k):
                self.trainingset = np.asarray(ts)
                self.tags = np.asarray(tags)
                self.classes = unique(tags)
                self.k = k

        def classify(self, X):
                dists = []
                for e in self.trainingset:
                    dists.append(np.linalg.norm(e-X))
                ordered = np.argsort(dists)
                kn = ordered[:self.k:]
                tagskn = self.tags[kn]
                tagskn.sort()
                ct = tagskn[0]
                maximum = 0
                counter = 0
                for i in range(len(tagskn)):
                    t = tagskn[i]
                    if i == 0 or t == ct:
                        counter += 1
                    else:
                        ct = t
                        counter = 1
                    if counter > maximum:
                        maximum = counter
                        tagmax = ct
                return int(tagmax)








