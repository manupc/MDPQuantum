#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:29:39 2023

@author: manupc

This file contains the classical ToyEnv and Quantum ToyEnv environments
used in the publication
"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, transpile
from qiskit.circuit.library import RYGate
from qiskit.circuit import Parameter
import numpy as np


# Implementation of classic MDP of the toy env in the publication
class ClassicToyEnv:
    
    # Constructor. 
    # Param MaxSteps: Maximum number of environment/agent interactions allowed for an episode
    def __init__(self, MaxSteps=np.Inf):
        
        self.MaxSteps= MaxSteps # Maximum number of steps before ending episode
        
        # ========================================
        # MDP definition
        self.nS= 4 # Number of MDP states
        self.nA= 2 # Number of MDP actions
        self.nR= 4 # Number of possible rewards

        # Transition function
        self.T= np.zeros((self.nS, self.nA, self.nS))
        self.T[0, 0, 1]= 0.7
        self.T[0, 0, 2]= 0.3
        self.T[0, 1, 1]= 1.0
        self.T[1, 1, 2]= 0.2
        self.T[1, 1, 3]= 0.8
        self.T[1, 0, 1]= 1.0
        self.T[2, 1, 0]= 0.4
        self.T[2, 1, 2]= 0.6
        self.T[2, 0, 1]= 0.2
        self.T[2, 0, 3]= 0.8
        self.T[3, 0, 2]= 1.0
        self.T[3, 1, 3]= 1.0

        # Reward function
        self.R= np.array([10, -5, 1, -10])
        
        # Current state ans step
        self.currentState= None
        self.currentStep= None
        
    
    # returns the number of states in the environment
    def numberOfStates(self):
        return self.nS
       
    # returns the number of states in the environment
    def numberOfActions(self):
        return self.nA
    
    # Returns the transition probability from state s and action a to state sp
    def transitionProb(self, s, a, sp):
        return self.T[s, a, sp]
    
    
    # Returns the reward value for the transition (s,a)->sp, i.e. r(s,a,sp)
    def rewardValue(self, s, a, sp):
        return self.R[sp]
    
        
    # Resets the environment to default state
    # Returns the current env. state
    def reset(self):
        
        # Set default env state and step number
        self.currentState= 0
        self.currentStep= 0
        return int(self.currentState)
        
    # Executes an action "action" over the environment
    # Returns the next state sp observation and reward r as  (sp, r)
    # Returns (None, None) if the Maximum number of steps criterion is True
    def step(self, action):
        
        # First: Check if the stopping criterion is True
        if self.StoppingCriterionSatisfied():
            return None, None
        
        # Calculate next state and reward
        sp= np.random.choice(range(self.nS), size=1, p=self.T[self.currentState, action].squeeze())
        r= self.R[sp]
        
        # Update step count and state
        self.currentState= sp
        self.currentStep+= 1
        
        return int(sp),r

    def StoppingCriterionSatisfied(self):
        return self.currentStep >= self.MaxSteps
    



# The ToyEnv classical environment with quantum implementation of 
# the underlying MDP
class QuantumToyEnv(ClassicToyEnv):
    
    # Constructor. 
    # Param MaxSteps: Maximum number of environment/agent interactions allowed for an episode
    def __init__(self, MaxSteps=np.Inf):
        super().__init__(MaxSteps)
    
        # ========================================
        # Quantum MDP representation 
        self.nqS= 2 # Number of qubits for state representation= log2(nS)
        self.nqA= 1 # Number of qubits required for action representation= log2(nA)
        self.nqR= 2 # Number of qubits for reward representation= log2(nR)


        # Quantum Registers    
        self.qS= QuantumRegister(self.nqS, name='qS') # Quantum Register to store input environment states
        self.qSp= QuantumRegister(self.nqS, name='qSp') # Quantum Register to store output environment states
        self.qA= QuantumRegister(self.nqA, name='qA') # Quantum Register to store actions
        self.qR= QuantumRegister(self.nqR, name='qR') # Quantum Register to store rewards
        
        # Classical Registers
        self.cS= ClassicalRegister(self.nqS, name='cS') # Classical Register to store measured next state
        self.cR= ClassicalRegister(self.nqR, name='cR') # Classical Register to store measured reward

        # Quantum Circuit to store a cycle of an MDP
        self.qc= QuantumCircuit(self.qS, self.qA, self.qSp, self.qR, self.cS, self.cR)
        
        
        # Us implementation: Appends Us to self.qc
        self.stateParameters= [ Parameter('s'+str(i)) for i in range(self.nqS) ]
        self.__Us(self.qc, self.qS, self.stateParameters)
        

        # Ua implementation: Appends Ua to self.qc
        self.actionParameters= [ Parameter('a'+str(i)) for i in range(self.nqA) ]
        self.__Ua(self.qc, self.qA, self.actionParameters)


        # Ut implementation: Appends transition function to self.qc
        self.__Ut(self.qc, self.qS, self.qA, self.qSp)        
                

        # Ur implementation: Appends Reward function to self.qc
        self.__Ur(self.qc, self.qSp, self.qR)
        
        
        # Measurement of next state
        self.qc.measure(self.qSp, self.cS)
        
        # Measurement of reward
        self.qc.measure(self.qR, self.cR)
        
        
        # Quantum simulator
        self.sim = Aer.get_backend('qasm_simulator')
        
        

    # Implementation of the QSample encoding
    # param probs: Probabilities of each amplitude
    # param qc: The quantum circuit where to append the encoding
    # param qS: Quantum register to store current env. state
    # param qA: Quantum register to store current action
    # param qSp: Quantum register to store next env. state
    # param control_s: str with values 0/1 to set the control state coming from qS
    # param control_a: str with values 0/1 to set the control state coming from qA
    # control_sp: Internal parameter to track recursion
    # currentQ: Internal parameter to track recursion
    #
    # The circuit qc is modified and the QSample is appender to the end
    def __QSampleNextState(self, probs, qc, qS, qA, qSp, control_s, control_a, control_sp= '', currentQ=0):
        
        all_controls= control_s + control_a + control_sp
        
        # Find control qubits
        controlsS= []
        for i in range(len(qS)):
            controlsS.append(qS[i])
        controlsA= []
        for i in range(len(qA)):
            controlsA.append(qA[i])
        controlsSp= []
        for i in range(currentQ):
            controlsSp.append(qSp[i])
        
        
        # Base case: End of recursion
        if currentQ>=len(qSp):
            return
        
        else: # General case
        
            # compute prob(qS[currentQ | qS, qA, qS[:currentQ-1]])
            add_prob= np.sum(probs)
            if add_prob > 0:
                current_prob= np.sum(probs[len(probs)//2:])/add_prob
                angle= 2*np.arcsin(np.sqrt(current_prob))
                
                # Make custom controlled CRY gate 
                cry= RYGate(angle).control(num_ctrl_qubits=len(all_controls), 
                                             ctrl_state= all_controls)
                
                qc.append(cry, [*controlsSp, *controlsA, *controlsS, qSp[currentQ]])
        
            self.__QSampleNextState(probs[:len(probs)//2], qc, qS, qA, qSp, 
                    control_s= control_s,
                    control_a= control_a,
                    control_sp=control_sp+'0', currentQ=currentQ+1)
            self.__QSampleNextState(probs[len(probs)//2:], qc, qS, qA, qSp, 
                    control_s= control_s,
                    control_a= control_a,
                    control_sp=control_sp+'1', currentQ=currentQ+1)
    

    # Circuit to set input environment state
    # param qc: The quantum circuit to be updated
    # param qS: Quantum register to store env state
    # param params: Parameters for the circuit
    # 
    # The qc is updated with the Us circuit at the end
    def __Us(self, qc, qS, params):
        for i in range(len(qS)):
            qc.rx(params[i]*np.pi, qS[i])
        qc.barrier()
        
    # Circuit to set input action
    # param qc: The quantum circuit to be updated
    # param qA: Quantum register to store agent action
    # param params: Parameters for the circuit
    # 
    # The qc is updated with the Ua circuit at the end
    def __Ua(self, qc, qA, params):
        for i in range(len(qA)):
            qc.rx(params[i]*np.pi, qA[i])
        qc.barrier()
        
    
    # Circuit to implement transition function
    # param qc: The quantum circuit to be updated
    # param qS: Quantum register to store env state
    # param qA: Quantum register to store agent action
    # param qSp: Quantum register to store env next state
    # 
    # The qc is updated with the Ut circuit at the end
    def __Ut(self, qc, qS, qA, qSp):
        
        for s in range(self.nS): 
            for a in range(self.nA):
                
                # Conditionals for input state s
                binS= bin(s)[2:][::-1]
                while len(binS)<self.nqS:
                    binS+= '0'
                binS= binS[::-1]
                
                # Conditionals for input action a
                binA= bin(a)[2:][::-1]
                while len(binA)<self.nqA:
                    binA+= '0'
                binA= binA[::-1]
        
                # Transition Function P(s,a)
                probs= self.T[s, a]
                self.__QSampleNextState(probs, self.qc, self.qS, self.qA, self.qSp, 
                        control_s=binS, 
                        control_a=binA)
                self.qc.barrier()

    
    # Circuit to implement reward function
    # param qc: The quantum circuit to be updated
    # param qSp: Quantum register to store env next state
    # param qR: Quantum register to store rewards
    # 
    # The qc is updated with the Ur circuit at the end
    def __Ur(self, qc, qSp, qR):
        
        # Hard-coded for example MDP
        for i in range(len(qSp)):
            qc.cx(qSp[i], qR[i])
        self.qc.barrier()

        
    # Executes an action "action" over the environment
    # Returns the next state sp observation and reward r as  (sp, r)
    # Returns (None, None) if the Maximum number of steps criterion is True
    def step(self, action):
        
        # First: Check if the stopping criterion is True
        if self.StoppingCriterionSatisfied():
            return None, None
        
        # Encode current state in variational Us circuit
        sValue= np.zeros(len(self.qS))
        counter= 0
        s= int(self.currentState)
        while s>0:
            sValue[counter]= (s & 1)
            s>>=1
            counter+= 1
        qc= self.qc.bind_parameters({self.stateParameters[i]:sValue[i] for i in range(len(self.stateParameters))})


        # Encode current action in variational Ua circuit
        aValue= np.zeros(len(self.qA))
        counter= 0
        a= int(action)
        while a>0:
            aValue[counter]= (a & 1)
            a>>=1
            counter+= 1
        qc= qc.bind_parameters({self.actionParameters[i]:aValue[i] for i in range(len(self.actionParameters))})


        # MDP cycle quantum simulation
        results = self.sim.run(transpile(qc, self.sim), shots=1).result()
        counts= results.get_counts()
        measurement= list(counts.keys())[0]
        
        # Get measurement for Sp state
        binsp= measurement[:-self.nqR][::-1]
        
        # Post processing: Calculate the sp state as integer
        sp= int(0)
        for bit in binsp:
            if bit == '1':
                sp|= 1
            sp<<= 1
        sp>>= 1 # Undo last shift
        
        
        # Get measurement for reward
        binr= measurement[-self.nqR:][::-1]
        
        # Post processing: Calculate the actual reward value
        r= int(0)
        for bit in binr:
            if bit == '1':
                r|= 1
            r<<= 1
        r>>= 1 # Undo last shift
        r= self.R[r]
        

        # Update step count and state
        self.currentStep+= 1
        self.currentState= sp
        
        return int(sp),r
    
    
