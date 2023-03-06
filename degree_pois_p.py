#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:18:36 2023

@author: wenkaix
"""


import numpy as np
import scipy
import scipy.stats as stat
import random

from itertools import combinations, combinations_with_replacement, product


import forest.benchmarking.distance_measures as dm


import matplotlib.pyplot as plt


n = 20

# k =3 # m = 100
k = 4

itr = 10000#0



nk = scipy.special.comb(n, k)
nk1 = scipy.special.comb(n-1, k-1)


# p_list = np.linspace(.1, .9, 9)

# p_list = np.linspace(.2, 1.6, 9)/nk1

p_list = np.linspace(.6, 2, 8)/nk1

index_set = list(combinations(range(n), k))

tv_dist = np.zeros(len(p_list))
tv_dist_emp = np.zeros(len(p_list))
# stein_err = p_list
stein_err = (nk1*p_list**2)
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



for p_id, prob in enumerate(p_list):

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
    
    tv_dist[p_id] = TVD_Poisson_exact(nk1*prob, deg)
    
print(tv_dist, stein_err)

# plt.plot(p_list, tv_dist_emp,color="yellow", marker="x", label="2Sample TV distance")
plt.plot(p_list, tv_dist,color="red", marker="x", label="Empirical TV distance")
plt.plot(p_list, p_list, color="blue", marker="^", label="Stein bound")
plt.plot(p_list, stein_err, color="black", marker="^", label="Stein bound small p")
plt.legend(prop={"size":13})
plt.xlabel("Edge probability", fontsize=15)
plt.savefig("TV_n"+str(n)+"_p_change_k"+str(k)+".pdf", bbox_inches='tight')
