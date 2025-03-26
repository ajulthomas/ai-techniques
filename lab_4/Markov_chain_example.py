# -*- coding: utf-8 -*-
"""
Markov Chains and Hidden Markov Model: An Overview
Based on: https://towardsdatascience.com/introduction-to-hidden-markov-models-cd2c93e6b781
Computing probabilties for a Markov chain


Assume a Markov Chain with three weather states, daily weather can correspond to the states 
{S1='snow',S2='rain',S3='sunshine'}. S={S1,S2,S3} denotes the state space
"""
import numpy as np
from numpy.linalg import matrix_power
import pandas as pd
import time

def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = np.zeros(T,dtype=int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi

"""vector of initial probabilities (pi), i.e., initial probability distribution of 
rv daily weather (X) starts with snow, rain or sunshine"""

pi = np.array([0, 0.2, 0.8]) 

"""State transition Matrix P. P(i,j) denotes probability that 
next state of rv X(t+1) = Sj, given rv is in current state X(t) = Si"""

""" See that rows Pi of P sum up to 1, while columns Pj do not"""
P = np.array([[0.3,0.3,0.4],[0.1,0.45,0.45],[0.2,0.3,0.5]])

""" The matrix P is also useful to know what is the probability of the rv transitioning to
   state Sj *t-time steps later* given that the current state is Si, mathematically, 
   P(Xt=Sj/X0=Si) or P(X(n+t)=Sj/Xn=Si). This is equal to P^{t}.
   Refer to https://www.stat.auckland.ac.nz/~fewster/325/notes/ch8.pdf"""
   
"""Given the probability distribution pi for the initial state of the rv X0, 
   and the state-transition probability matrix P, the probability distribution pi_t for
   Xt = (pi)'*P^{t}, where ' denotes the transpose operator. Vectors in general are
   column-based (nx1), and therefore pi' will be a row vector."""

""" For our problem, given the initial probability distribution (pd) pi, the probability distribution
for snow, rain and sun after 100 days would be"""

# pi_100 = np.dot(pi,matrix_power(P,100))

pi_50 = np.dot(pi, matrix_power(P, 50))
# print("On day 100, Prob. of Snow = %.3f, Rain = %.3f and Sunshine= %.3f"\
#       % (pi_100[0],pi_100[1],pi_100[2]))
# time.sleep(5) 

print(f'On day 50, Prob. of snow = {pi_50[0]:.3f}, Rain = {pi_50[1]:.3f}, Sunshine= {pi_50[2]:.3f}')
time.sleep(5) 

"""Transitioning from a Markov Chain to HMM, the difference is that the 
   states now become latent or unobservable. Suppose we only have a thermometer 
   in the house that ouputs HOT (H)/COLD (C) given the weather outside, we can end up with the 
   below pd."""
   
"""P(H/Snow) = 0,   P(C/Snow) = 1
      P(H/Rain) = 0.2, P(C/Rain) = 0.8
      P(H/Sunshine)=0.7, P(C/Sunshine)=0.3"""
      
"""Suppose we need to figure out the probability that it feels cold on
        two consecutive days, you have three possibilities for day 1: C/Snow, C/Rain and
        C/Sunshine, and similarly three for day 2. Totally 3x3 = 9 possibilities"""
        
"""One among the possibilities is that day1= Rain and day2 = Snow. 
       So we compute joint distribution for 
       P(C,C,Rain,Snow) = P(C,C/Rain,Snow)*P(Rain,Snow) 
       P(Rain,Snow) = P(Rain)*P(Snow/Rain) = 0.2*0.1 = 0.02 (we use pi as the pd for day 1)
       P(C,C/Rain,Snow) = P(C/Rain)*P(C/Snow) = 0.8*1 = 0.8
       Therefore, P(C,C,Rain,Snow) = 0.8*0.02 = 0.16"""
       
"""Likewise, P(C,C,Rain,Rain) = 0.2*0.45*0.8*0.8 = 0.64*0.09 < 0.06"""

"""Obviously, computing all 9 probabilities would be cumbersome"""
    
"""Given a sequence of H/C observations, and if we need to find the likely
      sequence of states that led to the observation sequence, as a brute-force method,
      we can determine which state-sequence maximizes the joint distibution. But this
      is computationally expensive with exponential complexity. Use Viterbi algorithm 
      instead"""


"""Demonstration of the Viterbi algorithm- Finding most likely sequence of hidden states
"""
obs_map = {'Cold':0, 'Hot':1}
# obs = np.array([1,1,0,1,0,0,1,0,1,1,0,0,0,1])
obs = np.array([0,0,1,0,1,1,0,1,0,0,1,1,1,0])

inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

print("Simulated Observations:\n",pd.DataFrame(np.column_stack([obs, obs_seq]),columns=['Obs_code', 'Obs_seq']) )
time.sleep(5) 


pi = [0.6,0.4] # initial probabilities vector
states = ['Cold', 'Hot']
hidden_states = ['Snow', 'Rain', 'Sunshine']
pi = [0, 0.2, 0.8]
state_space = pd.Series(pi, index=hidden_states, name='states')
a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
a_df.loc[hidden_states[0]] = [0.3, 0.3, 0.4]
a_df.loc[hidden_states[1]] = [0.1, 0.45, 0.45]
a_df.loc[hidden_states[2]] = [0.2, 0.3, 0.5]
print("\n HMM matrix:\n", a_df)
a = a_df.values

observable_states = states
b_df = pd.DataFrame(columns=observable_states, index=hidden_states)
b_df.loc[hidden_states[0]] = [1,0]
b_df.loc[hidden_states[1]] = [0.8,0.2]
b_df.loc[hidden_states[2]] = [0.3,0.7]
print("\n Emission matrix:\n",b_df)
b = b_df.values
time.sleep(5) 

path, delta, phi = viterbi(pi, a, b, obs)
state_map = {0:'Snow', 1:'Rain', 2:'Sunshine'}
state_path = [state_map[v] for v in path]
pd.DataFrame().assign(Observation=obs_seq).assign(Best_Path=state_path)

print("Most likely state path:\n",pd.DataFrame(np.column_stack([obs, obs_seq, state_path]),columns=['Obs_code', 'Obs_seq', 'State_seq']) )

"""Baum-Welch algorithm used to learn best-fit HMM parameters, and is similar to the EM algorithm"""