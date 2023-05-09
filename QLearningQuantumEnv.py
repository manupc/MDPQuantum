#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:29:39 2023

@author: manupc

This file contains an example of classic Q-Learning to calculate
the optimal deterministic policy of the agent in the toy environment of the article,
using the MDP implemented as a quantum circuit
"""
from environments import QuantumToyEnv
from algorithms import QLearning
import time




# Instantiate environment
env= QuantumToyEnv()


# Q-Learning Algorithm hyperparameters
gamma= 0.99 # Discount factor
MaxSteps= 200 # Maximum number of iterations
eps0= 0.5 # Initial epsilon value for e-greedy policy
epsf= 0.001 # Final epsilon value for e-greedy policy
epsSteps= MaxSteps # Number of steps to decrease e-greedy epsilon from eps0 to epsf
alpha= 0.2 # Learning rate

show= True # To show current step during simulation

# Execute Q-Learning algorithm
t0= time.time()
policy, Niter= QLearning(env, MaxSteps, eps0, epsf, epsSteps, alpha, gamma, show)
tf= time.time()


# Show results
print('Q-Learning stopped after {} iterations'.format(Niter))
print('The execution time was: {} s.'.format(tf-t0))
print('The policy obtained is:')
for s in range(len(policy)):
    print('\tFor state {}: Execute action {}'.format(s, policy[s]))
