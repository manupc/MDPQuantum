#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:29:39 2023

@author: manupc

This file contains an example of classic Value Iteration to calculate
the optimal deterministic policy of the agent in the toy environment of the article
"""
from environments import ClassicToyEnv
from algorithms import ValueIteration, ExtractPolicyFromVTable, runEnvironment
import time 
import numpy as np
import matplotlib.pyplot as plt



# Instantiate environment
env= ClassicToyEnv()


# Value-Iteration Algorithm hyperparameters
gamma= 0.99 # Discount factor
MaxIterations= 2000 # Maximum number of iterations
ConvThres= 1e-3 # Convergence threshold

# Execute Value-Iteration algorithm
t0= time.time()
Vtable, Niter, converged= ValueIteration(env, gamma, iterations=MaxIterations, convThres=ConvThres)

# Extract policy from Vtable
policy= ExtractPolicyFromVTable(env, Vtable, gamma)
tf= time.time()


# Execute policy
MaxIterations= 200 # Number of iterations for test
MaxTests= 100
RewardSet= []
for test in range(MaxTests):
    R, t_total= runEnvironment(env, policy, MaxIterations)
    RewardSet.append(R)
    

# Show results
print('Value Iteration stopped after {} iterations'.format(Niter))
print('The execution time was: {} s.'.format(tf-t0))
print('Value Iteration converged: {}'.format(converged))
print('The policy obtained is:')
for s in range(len(policy)):
    print('\tFor state {}: Execute action {}'.format(s, policy[s]))
print('Total Reward over {} experiences: {}'.format(MaxIterations, np.mean(RewardSet)))

plt.hist(RewardSet)
plt.xlabel('Total reward')
plt.ylabel('# Times obtained')
plt.title('Value Iteration')