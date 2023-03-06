#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:08:04 2023

@author: wenkaix
"""


import numpy as np
import scipy
import scipy.stats as stat
import random

from itertools import combinations, combinations_with_replacement, product

import matplotlib.pyplot as plt



def Stein_bound(nk, prob):
#     return (2. - (2.*prob**2*(1.-prob)**2 -1)/np.sqrt(prob*(1.-prob)))/np.sqrt(nk)
    return (2. + (prob**2. + (1-prob)**2.)/np.sqrt(prob*(1.-prob)))/np.sqrt(nk)



# n_list = [20, 50, 100, 200]
n_list = [20, 30, 50, 100, 200, 300]
prob = 0.003

k =3 # m = 100


itr = 1000#0

w_dist = np.zeros(len(n_list))

w_dist_emp = np.zeros(len(n_list))

for n_id, n in enumerate(n_list):

    nk = scipy.special.comb(n, k)
    nk1 = scipy.special.comb(n-1, k-1)
    
    index_set = list(combinations(range(n), k))

    deg = []
    for i in range(itr):
        np.random.seed(seed=i + 3663)
        s = np.random.binomial(nk, prob)
        hyperedgeList = random.sample(index_set, s)
        
        values, counts = np.unique(hyperedgeList, return_counts=True)
        # deg.extend(list(counts[np.random.choice(len(counts),10)]))
        deg.extend(list(counts))
        if i % 1000 == 0:
            print(len(values))
    deg = np.array(deg)
    deg_ = (deg - nk1*prob)/np.sqrt(nk1*prob*(1.-prob))
    deg_emp = (deg - np.mean(deg))/np.std(deg)

    # nsample = np.random.normal(nk1*prob, np.sqrt(nk1*prob*(1.-prob)), size=10000000)
    
    nsample = np.random.normal(0,1, size=10000000)
    w_dist[n_id] = stat.wasserstein_distance(nsample, deg_)
    # nsample_emp = np.random.normal(, size=10000000)
    w_dist_emp[n_id] = stat.wasserstein_distance(nsample, deg_emp)
   
    
    # nsample = np.random.normal(nk1*prob, np.sqrt(nk1*prob*(1.-prob)), size=15000000)
    # w_dist[n_id] = stat.wasserstein_distance(nsample, deg)
    # nsample_emp = np.random.normal(np.mean(deg), np.std(deg), size=15000000)
    # w_dist_emp[n_id] = stat.wasserstein_distance(nsample_emp, deg)
    


stein_err = np.zeros(len(n_list))
for n_id, n in enumerate(n_list):

    nk = scipy.special.comb(n, k)
    
    nk1 = scipy.special.comb(n-1, k-1)
    stein_err[n_id] = Stein_bound(nk1, prob) 

print(w_dist, w_dist_emp, stein_err)
# [  5.91085555  30.1004387   94.78796971 286.70599456] 
# [0.27473756 0.2636363  0.29601508 0.33511829] 
# [0.360145   0.13733201 0.06761761 0.033553  ]

#prob=0.6: [  8.09524274  17.06895856  41.29643304 129.99287806] [0.25164598 0.25496252 0.25361323 0.2724539 ] [0.29105901 0.1888931  0.11098785 0.05464664]
plt.plot(n_list, w_dist,color="red", marker="x", label="Empirical Wasserstein distance")
# plt.twinx()
# plt.plot(n_list, w_dist_emp, color="black", marker="x", label="Estimated Wasserstein distance")
# plt.twinx()
plt.plot(n_list, stein_err, color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size":13})
plt.xlabel("Size of hypergraph", fontsize=15)
plt.savefig("Wasserstein_n100_"+str(prob)+".pdf", bbox_inches='tight')

plt.show()

    
np.savez("./w-dist-p"+str(prob)+".npz",w_dist = w_dist, stein_err = stein_err, n_list = n_list, prob=prob)
