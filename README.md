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

The rule means that if a <img src="https://latex.codecogs.com/gif.latex?\theta^{'}" title="\theta^{'}" /> is more likely than the current <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />, then we always accept <img src="https://latex.codecogs.com/gif.latex?\theta^{'}" title="\theta^{'}" />. If it is less likely than the current <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />, then we might accept it or reject it. The less likely it is, the less probability we accept the new <img src="https://latex.codecogs.com/gif.latex?\theta^{'}" title="\theta^{'}" />.

In summary, the Metropolis-Hastings algorithm works as below.

Given
* <img src="https://latex.codecogs.com/gif.latex?f" title="f" />, the PDF of the distribution to sample from
* <img src="https://latex.codecogs.com/gif.latex?Q" title="Q" />, the transition model
* <img src="https://latex.codecogs.com/gif.latex?\theta_{0}" title="\theta_{0}" />, a first guess for <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />

Set
* <img src="https://latex.codecogs.com/gif.latex?\theta=\theta_{0}" title="\theta=\theta_{0}" />

For n iterations
* <img src="https://latex.codecogs.com/gif.latex?\theta^{'}=Q(\theta_{i})" title="\theta^{'}=Q(\theta_{i})" />
* <img src="https://latex.codecogs.com/gif.latex?ratio&space;=&space;\frac{p^{'}}{p}=\frac{f(D|\Theta=\theta^{'})P(\theta^{'})}{f(D|\Theta=\theta)P(\theta)}" title="ratio = \frac{p^{'}}{p}=\frac{f(D|\Theta=\theta^{'})P(\theta^{'})}{f(D|\Theta=\theta)P(\theta)}" />
* if ratio > 1
  * set <img src="https://latex.codecogs.com/gif.latex?\theta_{i}=\theta^{'}" title="\theta_{i}=\theta^{'}" />
* else
  * generate a uniform random number <img src="https://latex.codecogs.com/gif.latex?r" title="r" /> in <img src="https://latex.codecogs.com/gif.latex?[0,1]" title="[0,1]" />
  * if <img src="https://latex.codecogs.com/gif.latex?r<ratio" title="r<ratio" />, set <img src="https://latex.codecogs.com/gif.latex?\theta_{i}=\theta^{'}" title="\theta_{i}=\theta^{'}" />

## Hidden Markov Model (HMM)
HMM is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (hidden) states.

The term **hidden** refers to the first order Markov process behind the observation.
The term **observation** refers to the data we know and observe. It is shown by "Walk", "Shop" and "Clean" in the diagram below.
**Markov** process is shown by the interaction between "Rainy" and "Sunny" in the diagram below, and each of these are **hidden states**.

<p align="center">
  <img width="460" height="360" src="./weather_hmm.png">
</p>

Now let's define some variables of the model.

* <img src="https://latex.codecogs.com/gif.latex?T" title="T" /> = length of the observation sequence
* <img src="https://latex.codecogs.com/gif.latex?N" title="N" /> = number of states in the model
* <img src="https://latex.codecogs.com/gif.latex?M" title="M" /> = number of observation symbols
* <img src="https://latex.codecogs.com/gif.latex?Q=\left&space;\{&space;q_{0},&space;q_{1},...,q_{N-1}\right&space;\}" title="Q=\left \{ q_{0}, q_{1},...,q_{N-1}\right \}" /> = distinct states of the Markov process
* <img src="https://latex.codecogs.com/gif.latex?V=\left&space;\{&space;0,&space;1,...,M-1\right&space;\}" title="V=\left \{ 0, 1,...,M-1\right \}" /> = set of possible observations
* A = state transition probabilities (the arrows pointing to each hidden state)
* B = observation probability matrix (the blue and red arrows pointing to each observations from each hidden state; matrix is row stochastic meaning the rows add up to 1)
* <img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /> = initial state distribution (starts the model off with a hidden state)
* <img src="https://latex.codecogs.com/gif.latex?O=\left&space;\{&space;O_{0},&space;O_{1},...,O_{T-1}\right&space;\}" title="O=\left \{ O_{0}, O_{1},...,O_{T-1}\right \}" /> = observation sequence

In the diagram above, T=don't have any observation yet, N=2, M=3, Q={"Rainy", "Sunny"}, V= {"Walk", "Shop", "Clean"}. 

In the state transition matrix, 
<img src="https://latex.codecogs.com/gif.latex?a_{ij}=P(state\&space;q_{j}\&space;at\&space;t&plus;1\&space;|\&space;state\&space;q_{i}\&space;at\&space;t)" title="a_{ij}=P(state\ q_{j}\ at\ t+1\ |\ state\ q_{i}\ at\ t)" />
This matrix explains what the probability is from one state going to another.

In the observation probability matrix, 
<img src="https://latex.codecogs.com/gif.latex?b_{j}(k)=P(observation\&space;k\&space;at\&space;t\&space;|\&space;state\&space;q_{j}\&space;at\&space;t)" title="b_{j}(k)=P(observation\ k\ at\ t\ |\ state\ q_{j}\ at\ t)" />.
This matrix explains what the probability is from one state going to an observation.

The full model with known state transition probabilities, observation probability matrix, and initial state distribution is described as,
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\lambda=(A,B,\pi)" title="\lambda=(A,B,\pi)" />
</p>
