---
layout: post
title: "RL Course by David Silver"
date: 2018-12-14
categories: blog
tags: [RL,ML]
---
# Lecture 1: Introduction to Reinforcement Learning
RL using the reward assumption. It states that:
- **Reward hypothesis**: Every goal can be achieved by achieving a sequence of reward signal (expected cumulative reward)  

**Information State** (Markov property): contains all useful information from the history. -> P[St+1|St] = P[St+1|S1,...,St]. So we don't need to take account of the long tail of history (burden)  
*"The future is independent to the pass given the present"*

State:  
- **Fully Observable Environments**: When Se = Sa , Using MDP  
- **Partially Observable Environments**: Se ≠ Sa, using MOMDP  

## RL Agent Taxonomy
![](https://raw.githubusercontent.com/thesunkid19/blog/gh-pages/img/RLagenttaxonomy.jpg)
# Lecture 2: Markov Decision Processes
**Markov process** is a tuple <S,Pi>  
- S is a set of (finite) states  
- P is a state transition probability matrix add (Reward, discount value)  

**Markov Reward Process** is a tuple <S,P, R, γi>  = *Markov process* + reward information (R, γi)
- Rs = reward function (not just a scalar), $$ Rs = E[Rt+1 / St = s] $$,  because Rt+1 is a probability distribution.
- Return Gt is the sum of all reward from time-step t to the end of the enviroment (terminal state), usually with weight decay γ $$ Gt = E[Rt+1+γ*Rt+2+...|St = s] $$

**Markov Decision Process** is a tuple M = <S, A,P, R, γi> and a policy π  
- In this model, policy π is added for the choice of actions.

# Lecture 3: Planning by Dynamic programming.
There are 2 nice properties that DP is used for solving MDP optimal problem (because MDP fit to these properties):   
1. Optimal structure: break it down and solve subproblem = solving optimal problem
2. Overlapping subproblem: solving more fast by using cached, like using memory in Dynamic programming from programming algorithm.

There are 2 big task we want to solve in RL (by solving these tasks, we've solved the underlying task of RL - finding best policy to act):
1. Evaluate value function: to answer the question "under policy pi, what is our value evaluation" then
2. Control policy: optimise the value function by answering the question "what is the best policy?" (Control) 
## How to evaluate policy pi: update it with DP 
synchronous update iteration.
**Prediction problem**: random initial then iterative update the value function -> using Bellman expectation equation
**Control problem:**
- Policy Iteration -> using BEE + Greedy Policy wrt value Improvement
- Value Iteration -> using BEE and "optimal v(s) -> optimal v(s') with s' is precessor of s"

# Lecture 4: Model-free prediction
In the unknown MDP enviroment, we must use different evaluation schema &  control schema. 
## Monte Carlo
- Record the value of each path under the policy, then average them to estimate the value function of each state. 
- The trick is that we don't need to carry all the record by using **incremental mean** to update value function each step, and throw it after: 
$\begin{aligned} \mu _ { k } & =  \mu _ { k - 1 } + \frac { 1 } { k } \left( x _ { k } - \mu _ { k - 1 } \right) \end{aligned}$ z
from that, we apply to evaluate value function: 
$V \left( S _ { t } \right) \leftarrow V \left( S _ { t } \right) + \alpha \left( G _ { t } - V \left( S _ { t } \right) \right)$
with  $ \alpha = \frac { 1 } { N \left( S _ { t } \right) }$

- MC waits for the end of episode to determine the change (increase) of V(t)
- Notes: MC only works on the terminating enviroment & complete squences
## Temporal-difference learning (TD)
- TD exploit the Markov env & using Bellman equation by estimate V(St) in MC by V(St+1) + γRt+1 
$V \left( S _ { t } \right) \leftarrow V \left( S _ { t } \right) + \alpha \left( R _ { t + 1 } + \gamma V \left( S _ { t + 1 } \right) - V \left( S _ { t } \right) \right)$
- TD just need to wait only the next time step to update V(t)
- Intuition: We don't need to crash to update our value, we can imediately update from the estimate value of next state and using the observation reward in t+1 time step. 
- Approximate Vt+1 so it gives a bias on our Vt. But TD target is low variance than MC target because Gt value depends on a squence of enviroment & action. It's more efficient but sensitive to init value.
- Update guess towards a guess - Bootstraps 
- Both TD & MC sample pi while MDP doesn't
## TD(λ)
- TD(λ) generalizes TD for n-step return. Its spectrum is between TD and MC.
- Term λ is made for weighed return 
$G _ { t } ^ { \lambda } = ( 1 - \lambda ) \sum _ { k = 1 } ^ { \infty } \lambda ^ { k - 1 } G _ { t } ^ { ( k ) }$ 
$G _ { t } ^ { ( n ) } = R _ { t + 1 } + \gamma R _ { t + 2 } + \ldots + \gamma ^ { n } V \left( S _ { t + n } \right)$


 
**ζ**


	



 


