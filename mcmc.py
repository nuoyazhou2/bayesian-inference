#!/usr/bin/env python3

"""
Implement MCMC-Metropolis-Hastings from scratch

"""

import sys
import numpy as np

def prior(x):
    """ define prior """
    # x[0] = mu, x[1]=sigma (new or current)
    return 0 if x[1] <= 0 else 1 

def log_like_normal(x, data):
    """ compute the likelihood of the data given a sigma (new or current) """
    # x[0] = mu, x[1]=sigma (new or current)
    return np.sum(-np.log(x[1]*np.sqrt(2*np.pi))-((data-x[0])**2)/(2*x[1]**2))

def acceptance_rule(x, x_new):
    """ define whether to accept or reject the new value of parameter """
    if x_new > x:
        return True
    else:
        r = np.random.uniform(0, 1)
        ratio = np.exp(x_new-x) # we need to exponentiate it in order to compare to the random number
        # since we did log transformation
        return ratio > r

def transition_model(x):
    """ define transition model to walk from current sample to the new one """
    return [x[0], np.random.normal(x[1], 0.5, (1,))]

def metropolis_hastings(likelihood_computer, prior, transition_model, param_init,
                        iterations, data, acceptance_rule):
    """ build Metropolis Hastings algorithm
    likelihood_computer(x, data): returns the likelihood that these parameters generated the data
    transition_model(x): a function that draws a sample from a symmetric distribution and returns it
    param_init: a starting sample
    iterations: number of accepted to generated
    data: the data that we would like to model
    acceptance_rule(x, x_new): decides whether to accept or reject the new sample
    """
    x = param_init
    accepted = []
    rejected = []
    for i in range(iterations):
        x_new =  transition_model(x)
        x_like = likelihood_computer(x, data)
        x_new_like = likelihood_computer(x_new, data)
        if acceptance_rule(x_like + np.log(prior(x)), x_new_like + np.log(prior(x_new))):
            x = x_new
            accepted.append(x_new[1])
        else:
            rejected.append(x_new[1])
    return np.array(accepted), np.array(rejected)

def main():
    """ main """
    mod1=lambda t:np.random.normal(10, 3, t)
    population = mod1(30000) # Form a population of 30,000 individuals, with average=10 and scale=3
    observation = population[np.random.randint(0, 30000, 1000)] # Assume we only observe 1,000 of these individuals
    mu_obs = observation.mean()
    sd_init = 0.1
    iterations = 50000
    
    # We would like to find a distribution for sd using the 1000 observed samples
    # a. PDF: we take Gaussian as PDF since we find the 1000 samples are normally distributed after we made the histogram
    # b. transition distribution: we take Gaussian, as it's simple as a starting example
    # c. prior: the sd should be positive
    accepted, rejected = metropolis_hastings(log_like_normal, prior, transition_model, [mu_obs, sd_init],
                                             iterations, observation, acceptance_rule)
    
    print("{} samples are rejected.".format(len(rejected)))
    print("{} samples are accepted.".format(len(accepted)))
    print("The last 10 samples contain the following values for sd are:")
    print(accepted[-10:])

if __name__ == '__main__': main()

