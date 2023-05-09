#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:12:08 2023

@author: manupc


This file contains auxiliary functions to implement the Value Iteration and
the Q-Learning procedures. It assumes that the input environments
have the structure given in environments.py

"""

import numpy as np
from collections import defaultdict

###########################################################################
# VALUE ITERATION ALGORITHM
###########################################################################



# Algorithm: Value Iteration
# param env: The environment (with the internal structure given in environments.py)
# param gamma: Discount factor
# iterations: Number of iterations to stop the algorithm if convergence has not been achieved
# convThres: Convergence threshold to know whether the algorithm converges or not
# Returns a tuple (V, it, converge) containing:
#       V: the V(s) values as an array
#       it: The number of iterations performed by the algorithm
#       converge: Boolean to know if the algorithm has converged
def ValueIteration(env, gamma, iterations=20, convThres=1e-20):
    
    # Table containing V(s) values initialized to 0
    Vtable= np.zeros(env.numberOfStates())
    
    end= False # Loop Stopping criterion 
    converge= False # To know if the algorithm has converged
    
    
    it= 0 # Current iteration
    
    # Value Iteration main loop
    while not end: 
        auxVtable= Vtable.copy()
        
        # Update table Q(s,a)= sum_{s'} p(s'|s,a)*(r(s,a,s')+gamma*V(s))
        for s in range(env.numberOfStates()):
            Qtable= np.zeros(env.numberOfActions())
            
            for a in range(env.numberOfActions()):
                for sp in range(env.numberOfStates()):
                    
                    # Get transition probability from (s,a)->sp
                    p= env.transitionProb(s, a, sp)
                    if p>0.0: # Update Q-Table
                        Qtable[a]+= p*(env.rewardValue(s,a,sp) + gamma*auxVtable[sp] )
            
            # Update V(s)= max_{a} Q(s,a)
            Vtable[s]= np.max(Qtable)
        
        # Prepare next iteration
        it+= 1
        
        # Check convergence
        converge= np.max(np.fabs(Vtable-auxVtable)) <= convThres
        
        # Stopping criterion
        end= (it>=iterations or converge)
        
    #Return the V-table, the number of iterations performed and if the algorithm converged
    return Vtable, it, converge


# Method to extract a policy from a V(s) table
# param env: The environment
# param Vtable: An array containing the table V(s) for all states s
# param gamma: Discount factor
# 
# Returns policy, an array containing policy[s]= best action
def ExtractPolicyFromVTable(env, Vtable, gamma):
    
    # Create policy table
    policy= np.zeros(len(Vtable), dtype=int) 
    
    for s in range(len(Vtable)):
        
        # Create Q-table
        Qtable= np.zeros(env.numberOfActions())
        for a in range(env.numberOfActions()):
            for sp in range(env.numberOfStates()):
                
                # Get transition probability from (s,a)->sp
                p= env.transitionProb(s, a, sp)
                if p>0.0:
                    Qtable[a]+= p*(env.rewardValue(s,a,sp) + gamma*Vtable[sp] )

        # Update policy
        policy[s]= np.argmax(Qtable)
        
    return policy

    




###########################################################################
# Q-LEARNING ALGORITHM
###########################################################################



# e-greedy policy implementation
# param env: The environment
# param S: The current state observation
# param AgentPolicy: A callable object AgentPolicy(env, S, Q) containing the agent policy
# param Q: the Q-table Q[(s,a)]-> Value of state-action pair (s,a)
# param epsilon: The value epsilon for the e-greedy policy
#
# Returns an action to be executed in the environment
def eGreedyPolicy(env, S, AgentPolicy, Q, epsilon):
    
    if np.random.rand() < epsilon: # Random uniform policy
        return np.random.randint(low= 0, high=env.numberOfActions()) 
    
    else:
        return AgentPolicy(env, S, Q)
    

# Agent Policy that selects the action with maximum Q-value
# param env: The environment
# param S: The current state observation
# param Q: the Q-table Q[(s,a)]-> Value of state-action pair (s,a)
#
# Returns an action to be executed in the environment
def AgentPolicy(env, S, Q):
    return np.argmax([Q[(S, a)] for a in range(env.numberOfActions())]) 





# Q-Learning algorithm with epsilon-greedy policy with epsilon linear decrease
# param env: The environment
# param MaxSteps: Maximum number of env. steps to run
# param eps0: Initial value for epsilon
# param epsf: Final value for epsilon
# epsEpisodes: Number of env steps to decrease epsilon from eps0 to epsf
# alpha: Learning rate
# gamma: Discount factor
# show: True to print on console the current iteration
# Returns a tuple (policy, it, converge) containing:
#       policy: the deterministic policy policy[s] for all s
#       it: The number of iterations performed by the algorithm
#       converge: Boolean to know if the algorithm has converged
def QLearning(env, MaxSteps, eps0, epsf, epsSteps, alpha, gamma, show=False):
    
    # Initialize epsilon
    epsilon= eps0
    
    # Initialize Q-table  Q(s,a)= 0 
    Qini= defaultdict(float)
    
    end= False # Stopping criterion
    it= 0 # Number of steps performed
    
    # Initialize environment and get initial state
    s= env.reset()
    
    # Q-Learning cycle
    while not end:

        if show:
            print('Running step {}'.format(it+1))

        Q= Qini.copy()
        
        # Select action for current state
        a= eGreedyPolicy(env, s, AgentPolicy, Q, epsilon)
        
        # Execute action in environment
        sp, r= env.step(a)
        
        # Get best known action ap
        ap= np.argmax( [Q[(sp, a_)] for a_ in range(env.numberOfActions())] ) 
        
        # Q-Learning update rule
        Q[(s,a)]+= alpha*(r+gamma*Q[(sp, ap)] - Q[(s,a)])


        # Prepare next step
        s= sp


        # Reset env if required
        if env.StoppingCriterionSatisfied():
            sp= env.reset()
        
        # Update epsilon
        epsilon= max(epsf, eps0+it*(epsf-eps0)/epsSteps)
        
        # prepare next iteration
        Qini= Q
        it+= 1
        if it>=MaxSteps:
            end= True
            
            
    # calculate policy
    policy= np.zeros(env.numberOfStates(), dtype=int) 
    for s in range(env.numberOfStates()):
        action= int(AgentPolicy(env, s, Qini))
        policy[s]= action
    
    return policy, it