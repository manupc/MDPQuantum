# MDPQuantum
Implementation of a Markov Decision Process (MDP) into a quantum program targeted at reinforcement learning

This repository contains the source code for the article "Implementation of Markov Decision Processes into quantum algorithms for reinforcement learning", submitted to the 1st Workshop on Quantum Artificial Intelligence (https://quasar.unina.it/qai2023.html)

If you plan to use this code, please cite the following source:

M.P. Cuellar, M.C. Pegalajar, L.G.B. Ruiz, G. Navarro, C. Cano, L. Servadei, "Implementation of Markov Decision Processes into quantum algorithms for reinforcement learning", 1st Workshop on Quantum Artificial Intelligence 2023 (submitted)




The file environments.py contain a toy example of a MDP embedded into a classical environment (class ClassicToyEnv) and a quantum environment (class QuantumToyEnv).

The file algorithms.py contain the implementation of the classic Value Iteration and Q-Learning algorithms, adapted to use the environments in environments.py.

The files ClassicValueIteration.py and ClassicQLearning.py contain examples to use the Value Iteration and Q-Learning classic methods over the classic environment.

The file QLearningQuantum.py contains examples to use the Q-Learning procedure over the quantum environment.

