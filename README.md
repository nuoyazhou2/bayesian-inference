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
MCMC is a class of techniques for sampling from a probability distribution. It can be used to compute the distribution over the parameters given a set of observations and a prior belief. 

### Metropolis-Hastings
Metropolis-Hastings is a specific implementation of MCMC. It works well in high dimensional spaces as opposed to Gibbs sampling and rejection sampling. 

Metropolis-Hastings requires a simple distribution called the **proposal distribution** <img src="https://latex.codecogs.com/gif.latex?Q(\theta^{'}|\theta)" title="Q(\theta^{'}|\theta)" /> to draw samples from the posterior distribution. This technique uses the proposal distribution to randomly walk in the distribution space, accepting or rejecting jumps to new positions based on how likely the sample is. This "memoriless" random walk is the **Markov Chain** part of MCMC.

The **likelihood** of each new sample is determined by a function <img src="https://latex.codecogs.com/gif.latex?f" title="f" />. It is commonly chosen to be a probability density function that expresses the proportionality to the posterior we want to sample from.

Here is how the random walking works. We take our current parameter <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />, and propose a new one <img src="https://latex.codecogs.com/gif.latex?\theta^{'}" title="\theta^{'}" /> which is a random sample drawn from <img src="https://latex.codecogs.com/gif.latex?Q(\theta^{'}|\theta)" title="Q(\theta^{'}|\theta)" />. Often this is a symmetric distribution (e.g. Gaussian). To decide if <img src="https://latex.codecogs.com/gif.latex?\theta^{'}" title="\theta^{'}" /> is to be accepted or rejected, we calculate a ratio for each new proposed <img src="https://latex.codecogs.com/gif.latex?\theta^{'}" title="\theta^{'}" />.

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\frac{P(\theta^{'}|D)}{P(\theta|D)}&space;=&space;\frac{P(D|\theta^{'})P(\theta^{'})}{P(D|\theta)P(\theta)}" title="\frac{P(\theta^{'}|D)}{P(\theta|D)} = \frac{P(D|\theta^{'})P(\theta^{'})}{P(D|\theta)P(\theta)}" />
</p>

Which is equivalent to:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\frac{\prod_{i}^{n}f(d_{i}|\Theta=\theta^{'})P(\theta^{'})}{\prod_{i}^{n}f(d_{i}|\Theta=\theta)P(\theta)},&space;\textup{where}\&space;f\&space;\textup{is\&space;the\&space;proportional\&space;function\&space;mentioned\&space;above}." title="\frac{\prod_{i}^{n}f(d_{i}|\Theta=\theta^{'})P(\theta^{'})}{\prod_{i}^{n}f(d_{i}|\Theta=\theta)P(\theta)}, \textup{where}\ f\ \textup{is\ the\ proportional\ function\ mentioned\ above}." />
</p>

The rule for acceptance can then be formulated as:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?P(accept)&space;=\begin{cases}&space;\frac{\prod_{i}^{n}f(d_{i}|\Theta=\theta^{'})P(\theta^{'})}{\prod_{i}^{n}f(d_{i}|\Theta=\theta)P(\theta)},&space;&&space;\prod_{i}^{n}f(d_{i}|\Theta=\theta^{'})P(\theta^{'})&space;<&space;\prod_{i}^{n}f(d_{i}|\Theta=\theta)P(\theta)&space;\\&space;1,&space;&&space;\prod_{i}^{n}f(d_{i}|\Theta=\theta^{'})P(\theta^{'})&space;\geq&space;\prod_{i}^{n}f(d_{i}|\Theta=\theta)P(\theta)&space;\end{cases}$$" title="P(accept) =\begin{cases} \frac{\prod_{i}^{n}f(d_{i}|\Theta=\theta^{'})P(\theta^{'})}{\prod_{i}^{n}f(d_{i}|\Theta=\theta)P(\theta)}, & \prod_{i}^{n}f(d_{i}|\Theta=\theta^{'})P(\theta^{'}) < \prod_{i}^{n}f(d_{i}|\Theta=\theta)P(\theta) \\ 1, & \prod_{i}^{n}f(d_{i}|\Theta=\theta^{'})P(\theta^{'}) \geq \prod_{i}^{n}f(d_{i}|\Theta=\theta)P(\theta) \end{cases}$$" />
</p>

Note the prior components are often crossed out if there is no preference or restriction on the parameters.



