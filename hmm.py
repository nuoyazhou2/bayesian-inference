#!/usr/bin/env python3

"""
Apply HMM to solve a modeling problem

Assume the following conditions,

states = ('Rainy', 'Sunny')
 
observations = ('walk', 'shop', 'clean')
 
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
 
transition_probability = {
   'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
   'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
}
 
emission_probability = {
   'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
   'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
}

"""

import math
import numpy as np
from hmmlearn import hmm

def calc_prob_of_seq(model, *index):
    """ calculate the probability of an observed sequence happening, given a known model """
    # note that index is a tupple e.g. (0,), (0,1)
    assert all(isinstance(x, int) for x in index), "Some indices are not of integer type"
    log_prob = model.score(np.array([index]))
    prob = math.exp(log_prob)
    obs_seq_str = "-".join([observations[i] for i in index])
    print("The probability of observing \"{}\" is {:.3f}".format(obs_seq_str, prob))

def calc_optimal_hidden_states(model, *index):
    """ calculate the optimal hidden state sequence, given a known model and an observed sequence happening """
    assert all(isinstance(x, int) for x in index), "Some indices are not of integer type"
    logprob, seq = model.decode(np.array([index]).transpose())
    optimal_state_seq_str = "-".join([states[i] for i in seq])
    prob = math.exp(logprob)
    obs_seq_str = "-".join(observations[i] for i in index)
    print("Given the observations \"{}\", the optimal sequence of hidden states is \"{}\", with probability {:.3f} ".format(
    obs_seq_str, optimal_state_seq_str, prob))

def main():
    """ main """
    model = hmm.MultinomialHMM(n_components=2)
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.7, 0.3],
                                [0.4, 0.6]])
    model.emissionprob_ = np.array([[0.1, 0.4, 0.5],
                                    [0.6, 0.3, 0.1]])
    
    # 1. calculate the probability of an observed sequence happening
    calc_prob_of_seq(model, 0) # walk
    calc_prob_of_seq(model, 1) # shop
    calc_prob_of_seq(model, 2) # clean
    calc_prob_of_seq(model, 2, 2, 2) # clean-clean-clean
    
    # 2. calculate the optimal hidden state sequence, given an observed sequence 
    calc_optimal_hidden_states(model, 1, 2, 0) # shop-clean-walk
    calc_optimal_hidden_states(model, 2, 2, 2) # clean-clean-clean
    
if __name__ == '__main__':
    states = ("Rainy", "Sunny")
    observations = ("walk", "shop", "clean")
    main()