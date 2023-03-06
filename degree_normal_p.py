#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 08:33:26 2023

@author: wenkaix
"""


import numpy as np
import scipy
import scipy.stats as stat
import random

from itertools import combinations, combinations_with_replacement, product

import matplotlib.pyplot as plt



def Stein_bound(nk, prob):
    # return (2. - (2.*prob**2*(1.-prob)**2 -1)/np.sqrt(prob*(1.-prob)))/np.sqrt(nk)
    return (2. + (prob**2. + (1-prob)**2.)/np.sqrt(prob*(1.-prob)))/np.sqrt(nk)


n = 50

k =3 # m = 100

p_list = np.linspace(0.1, 0.9, 9)


itr = 1000#0

w_dist = np.zeros(len(p_list))

w_dist_emp = np.zeros(len(p_list))


nk = scipy.special.comb(n, k)
nk1 = scipy.special.comb(n-1, k-1)

index_set = list(combinations(range(n), k))


stein_err = np.zeros(len(p_list))
for p_id, prob in enumerate(p_list):
    nk1 = scipy.special.comb(n-1, k-1)
    stein_err[p_id] = Stein_bound(nk1, prob) 
    
    
for p_id, prob in enumerate(p_list):

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
    w_dist[p_id] = stat.wasserstein_distance(nsample, deg_)
    # nsample_emp = np.random.normal(, size=10000000)
    w_dist_emp[p_id] = stat.wasserstein_distance(nsample, deg_emp)
    



print(w_dist, w_dist_emp, stein_err)

plt.plot(p_list, w_dist,color="red", marker="x", label="Empirical Wasserstein distance")
# plt.plot(p_list, w_dist_emp, color="black", marker="o", label="Estimated Wasserstein distance")
plt.plot(p_list, stein_err, color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size":13})
plt.xlabel("Edge probability", fontsize=15)
plt.savefig("Wasserstein_n"+str(n)+"_p_change.pdf", bbox_inches='tight')




plt.plot(p_list, w_dist,color="red", marker="x", label="Empirical Wasserstein distance")
plt.plot(p_list, w_dist_emp, color="black", marker="o", label="Estimated Wasserstein distance")

plt.legend(prop={"size":13})
plt.xlabel("Edge probability", fontsize=15)
plt.savefig("normalise_n"+str(n)+"_p_change_no_upperbound.pdf", bbox_inches='tight')