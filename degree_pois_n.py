#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 19:37:07 2023

@author: wenkaix
"""

import numpy as np
import scipy
import scipy.stats as stat
import random

from itertools import combinations, combinations_with_replacement, product


import forest.benchmarking.distance_measures as dm


import matplotlib.pyplot as plt




itr = 100000


# n_list = [20, 50, 100, 200]
# n_list = [10, 20, 30, 50, 100, 150]
n_list = [10, 20, 30, 40, 50, 60]

k =4 # m = 100

tv_dist = np.zeros(len(n_list))
tv_dist_emp = np.zeros(len(n_list))
p_list = np.zeros(len(n_list))
# stein_err - p_list
      
def total_variation_distance(s1, s2):
    """
    s1, s2: np.arrays of samples
    """
    s_all = np.concatenate((s1,s2))
    values, counts = np.unique(s_all, return_counts=True)
    s1_ = np.concatenate((s1,values))
    v1, c1 = np.unique(s1_, return_counts=True)
    c1 -= 1
    c2 = counts - c1
    d1 = c1/np.sum(c1)
    d2 = c2/np.sum(c2)    
    return 0.5*np.sum(abs(d1-d2))


def TVD_Poisson_exact(lam, sample):
    """
    lam: Poisson parameter 
    sample: np.arrays of samples
    """
    values, counts = np.unique(sample, return_counts=True)
    P = stat.poisson.pmf(values, lam)
    P /= np.sum(P)
    Q = counts/np.sum(counts)
    return 0.5*np.sum(abs(P-Q))

c=1.

for n_id, n in enumerate(n_list):

    nk = scipy.special.comb(n, k)
    nk1 = scipy.special.comb(n-1, k-1)
    prob = c/float(nk1)
    p_list[n_id] = prob
    index_set = list(combinations(range(n), k))

    deg = []
    for i in range(itr):
        np.random.seed(seed=i + 3663)
        s = np.random.binomial(nk, prob)
        hyperedgeList = random.sample(index_set, s)
        values, counts = np.unique(hyperedgeList, return_counts=True)
        # deg.extend(list(counts[np.random.choice(len(counts),10)]))
        deg.extend(list(counts))
        if i % 10000 == 0:
            print(len(values))
    deg.extend(np.zeros(itr * n - len(deg)))
    deg = np.array(deg)
    # nsample = np.random.poisson(nk1*prob, size=1000000)
    # # nsample = nsample[nsample > 0]
    # tv_dist_emp[p_id] = total_variation_distance(nsample, deg)
    
    tv_dist[n_id] = TVD_Poisson_exact(nk1*prob, deg)

stein_err = (nk1*p_list**2)
    
print(tv_dist, stein_err)

plt.figure()
# plt.plot(n_list, tv_dist_emp,color="yellow", marker="x", label="2Sample TV distance")
plt.plot(n_list, tv_dist,color="red", marker="x", label="Empirical TV distance")
plt.plot(n_list, p_list, color="blue", marker="^", label="Stein bound")
# plt.plot(n_list, stein_err, color="black", marker="^", label="Stein bound small p")
plt.legend(prop={"size":13})
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.savefig("TV_c"+str(c)+"_n_change_k"+str(k)+".pdf", bbox_inches='tight')

#plot log

plt.figure()
plt.plot(n_list[:-1], np.log(tv_dist[:-1]),color="red", marker="x", label="Empirical TV distance")
plt.plot(n_list[:-1], np.log(p_list[:-1]), color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size":13})
plt.xlabel("Size of the hypergraph", fontsize=15)
plt.ylabel("Log distance", fontsize=15)
plt.savefig("LogTV_c"+str(c)+"_n_change_k"+str(k)+".pdf", bbox_inches='tight')
