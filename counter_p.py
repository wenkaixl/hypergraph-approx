#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:22:06 2023

@author: wenkaix
"""


import numpy as np
import scipy
import scipy.stats as stat
import random

from itertools import combinations, combinations_with_replacement, product


import matplotlib.pyplot as plt



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


def Stein_bound(nk, prob):
    # return (2. - (2.*prob**2*(1.-prob)**2 -1)/np.sqrt(prob*(1.-prob)))/np.sqrt(nk)
    return (2. + (prob**2. + (1-prob)**2.)/np.sqrt(prob*(1.-prob)))/np.sqrt(nk)


n = 30

k =3 

itr = 10000


nk = scipy.special.comb(n, k)
nk1 = scipy.special.comb(n-1, k-1)

p_list = np.linspace(0.1, 0.9, 9)

index_set = list(combinations(range(n), k))

tv_dist = np.zeros(len(p_list))

stein_err = (nk1*p_list**2)
# stein_err - p_list



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
    tv_dist[p_id] = TVD_Poisson_exact(nk1*prob, deg)

plt.plot(p_list, tv_dist,color="red", marker="x", label="Empirical TV distance")
plt.plot(p_list, p_list, color="blue", marker="^", label="Stein bound")
# plt.plot(p_list, stein_err, color="black", marker="^", label="Stein bound small p")
plt.legend(prop={"size":13})
plt.xlabel("Edge probability", fontsize=15)
plt.savefig("Poisson_big_p_n"+str(n)+".pdf", bbox_inches='tight')





p_list = np.linspace(.6, 2, 8)/nk1
w_dist = np.zeros(len(p_list))


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
        if i % 10000 == 0:
            print(len(values))
    deg = np.array(deg)
    deg_ = (deg - nk1*prob)/np.sqrt(nk1*prob*(1.-prob))
    # deg_emp = (deg - np.mean(deg))/np.std(deg)

    # nsample = np.random.normal(nk1*prob, np.sqrt(nk1*prob*(1.-prob)), size=10000000)
    
    nsample = np.random.normal(0,1, size=1000000)
    w_dist[p_id] = stat.wasserstein_distance(nsample, deg_)



plt.plot(p_list, w_dist,color="red", marker="x", label="Empirical Wasserstein distance")
plt.plot(p_list, stein_err, color="blue", marker="^", label="Stein bound")
plt.legend(prop={"size":13})
plt.xlabel("Edge probability", fontsize=15)
plt.savefig("Normal_small_p_example_n"+str(n)+".pdf", bbox_inches='tight')    
