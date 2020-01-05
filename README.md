# Bayesian Inference

>When you want to use a new algorithm that you don’t deeply understand, the best approach is to implement it yourself to learn how it works, and then use a library to benefit from robust code. -- Hilary Mason

Let's implement from scratch: Bayesian Inference, Markov Chain, Hidden Markov Model (HMM), Markov Chain Monte Carlo (MCMC), in Python.

## Frequentist vs Bayesian
There are two interpretations to probabilities.

| Frequentist | Bayesian | 
| ------------- |-------------|
| probabilities represent long term frequencies with which events occur | probabilities are treated as an expression of belief |
| there is no belief in a frequentist’s view of probability | The idea behind Bayesian thinking is to keep updating the beliefs as more evidence is provided |
| referred to as the objective approach since there is no expression of belief and/or prior events in it | referred to as the subjective view on probability as it deals with belief |
| Frequentists will never say “I am 50% (0.5) sure that there is rain today” | it is perfectly reasonable for a Bayesian to say “I am 50% (0.5) sure that there is rain today” |

## Bayesian Inference

Bayes' theorem describes what the probability that even A occurs is given some event B. 

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?P(A|B)&space;=&space;\frac{P(B|A)P(A)}{P(B)}" title="P(A|B) = \frac{P(B|A)P(A)}{P(B)}" />
</p>

With:
* P(A|B) is the **posterior** (what we wish to compute)
* P(B|A) is the **likelihood** (how likely is B assuming A occurred)
* P(A) is the **prior** (how likely is A regardless of evidence)
* P(B) is the **evidence** (how likely is the evidence)

Here, we are mostly interested in the specific formulation of Bayes' formula>
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?P(\theta|D)&space;=&space;\frac{P(D|\theta)P(\theta)}{P(D)},&space;\textup{where}\&space;P(\theta|D)\&space;\textup{is}\&space;\textup{the}\&space;\mathbf{posterior},\&space;P(D|\theta)\&space;\textup{is}\&space;\textup{the}\&space;\mathbf{likelihood},\&space;P(\theta)\&space;\textup{is}\&space;\textup{the}\&space;\mathbf{prior},\&space;and\&space;P(D)\&space;\textup{is}\&space;\textup{the}\&space;\mathbf{evidence}." title="P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}, \textup{where}\ P(\theta|D)\ \textup{is}\ \textup{the}\ \mathbf{posterior},\ P(D|\theta)\ \textup{is}\ \textup{the}\ \mathbf{likelihood},\ P(\theta)\ \textup{is}\ \textup{the}\ \mathbf{prior},\ and\ P(D)\ \textup{is}\ \textup{the}\ \mathbf{evidence}." />
</p>

MCMC allows us to sample from the posterior, and draw distributions over our parameters without having to worry about computing **evidence**.

## Markov Chain Monte Carlo (MCMC)
MCMC is a class of techniques for sampling from a probability distribution. It can be used to estimate the distribution of parameters given a set of observations.

