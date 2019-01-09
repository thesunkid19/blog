---
layout: post
title: "RL Course by David Silver"
date: 2018-12-14
categories: blog
tags: [RL,ML]
---
**Warning: This is a rant post containing a bunch of unorganized thoughts yet. This will be used as a resource for more structured posts later**

> “If people do not believe that mathematics is simple, it is only because they do not realize how complicated life is.” — John von Neumann
This course shows me a lot about how complicated the world is.

# Lecture 1: Introduction to Reinforcement Learning
RL using the reward assumption. It states that:
- **Reward hypothesis**: Every goal can be achieved by achieving a sequence of reward signal (expected cumulative reward)  

**Information State** (Markov property): contains all useful information from the history. -> $P[St+1|St] = P[St+1|S1,...,St]$. So we don't need to take account of the long tail of history (burden)  
*"The future is independent to the pass given the present"*

State:  
- **Fully Observable Environments**: When $Se = Sa$ , Using MDP  
- **Partially Observable Environments**: $Se ≠ Sa$, using MOMDP  

## RL Agent Taxonomy
![](https://raw.githubusercontent.com/thesunkid19/blog/gh-pages/img/RLagenttaxonomy.jpg)
# Lecture 2: Markov Decision Processes
**Markov process** is a tuple $<S,Pi>$  
- $S$ is a set of (finite) states  
- $P$ is a state transition probability matrix add (Reward, discount value)  

**Markov Reward Process** is a tuple $<S,P, R, γi>$  = *Markov process* + reward information $(R, γi)$
- $Rs$ = reward function (not just a scalar), $$ Rs = E[Rt+1 / St = s] $$,  because Rt+1 is a probability distribution.
- Return $Gt$ is the sum of all reward from time-step t to the end of the enviroment (terminal state), usually with weight decay γ $$ Gt = E[Rt+1+γ*Rt+2+.../ St = s] $$

**Markov Decision Process** is a tuple $M = <S, A,P, R, γi>$ and a policy π  
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
In the unknown MDP enviroment, we must use different evaluation schema &  control schema. \\
## Monte Carlo
- Record the value of each path under the policy, then average them to estimate the value function of each state. 
- The trick is that we don't need to carry all the record by using **incremental mean** to update value function each step, and throw it after: 
$\begin{aligned} \mu _ { k } & =  \mu _ { k - 1 } + \frac { 1 } { k } \left( x _ { k } - \mu _ { k - 1 } \right) \end{aligned}$ z
from that, we apply to evaluate value function: 
$V \left( S _ { t } \right) \leftarrow V \left( S _ { t } \right) + \alpha \left( G _ { t } - V \left( S _ { t } \right) \right)$
with  $ \alpha = \frac { 1 } { N \left( S _ { t } \right) }$

- MC waits for the end of episode to determine the change (increase) of V(t)
- Notes: MC only works on the terminating enviroment & complete squences \\
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
- Term λ is made for weighed return \\
$G _ { t } ^ { \lambda } = ( 1 - \lambda ) \sum _ { k = 1 } ^ { \infty } \lambda ^ { k - 1 } G _ { t } ^ { ( k ) }$ 
$G _ { t } ^ { ( n ) } = R _ { t + 1 } + \gamma R _ { t + 2 } + \ldots + \gamma ^ { n } V \left( S _ { t + n } \right)$

# Lecture 5: Model-Free Control
## On-policy learning
Using MC method can make agent get stuck on Greedy action selection by restricting from exploring another solution with lower initial value.

A simple idea is proposed to solve this: Take an $\epsilon$ probability to take action randomly and otherwise take the greedy action in $1 - \epsilon$  probability, so then we have a distribution:
$$\pi ( a | s ) = \left\{ \begin{array} { l l } { \epsilon / m + 1 - \epsilon } & { \text { if } a ^ { * } = \underset { a \in \mathcal { A } } { \operatorname { argmax } } Q ( s , a ) } \\ { \epsilon / m } & { \text { otherwise } } \end{array} \right.$$

 It's  **$\epsilon -greedy$ policy improvement**

**(Greedy in the Limit with Infinite Exploration) GLIE Monte-Carlo** method help building a model that still act $\epsilon -greedy$ but also solve the problem of converge stage when we've found optimal action-value func and don't want to act $\epsilon -greedy$, by using:
1/ Updating by every episode
2/ Limit with Infinite Exploration (set $\epsilon = 1/k$ then it'll converge to 0 at the end)

The different between MC control & TD control is MC uses `every episode` update scheme while TD uses `every time-step` update scheme (what is the different??). TD control using SARSA policy evaluation.

The sharing concept of 2 algorithm is (like $TD(\lambda)$) SARSA n-step. The Eligibility trace using in SARSA($\lambda$) is a kind of dimunitival sum by time that focus more on the latter time-step of anepisode than the first one.

## Off-policy learning
This strategy is often used for learning from observing humans or other agents (imitate learning), or exploring policy purpose to learn final optimal policy.

We can use the idea Importance Sampling ($$\mathbb { E } _ { X \sim P } [ f ( X ) ] = \sum P ( X ) f ( X ) = \mathbb { E } _ { X \sim Q } \left[ \frac { P ( X ) } { Q ( X ) } f ( X ) \right]$$) to apply to Off-policy by using a target generated from a policy $\mu$ to evaluate $\pi$. But it seems not work well with MC because of the long tail of step cause the ... something (high variance ??) 

**Q-learning** is a off-policy control method that no require importance sampling to find the optimal action-value function. It looks like BOE but using sampling technique. Not fully understand it yet.

Here is the summary all algorithm in DP vs TD.
![](https://raw.githubusercontent.com/thesunkid19/blog/gh-pages/img/DPvsTD.png)

Notes: On-policy & off-policy is quite hard to fully understand from Silver lecture, I just find a clear explanation about these terms in [this topic](https://www.quora.com/What-is-a-simple-iterative-example-of-how-on-policy-and-off-policy-algorithms-differ-in-reinforcement-learning-I-really-need-to-see-it-working-for-at-least-3-steps). That said, in RL, there are 2 terms that rarely is mentioned explicitly is "learning policy and behaviour policy: the former is which values you are trying to learn about and the latter is the policy that you follow to choose actions and get samples from".

- In on-policy: learning and behaviour policies are identical.
- In off-policy (Q-learning): the learning policy is greedy policy and the behaviour policy is something like $\epsilon$ greedy policy.

# Lecture 6: Model-Free Control

RL is known as an generalized method to solve large problems. The strategy to build value function approximation it uses is not just storage information/a lookup table but efficient strategies to representing and learning state.

**Question:**
- why SGD always converge on global optimum? 
- And oh wow, why look up table is a special case of value function approximate?

The idea of linear combination state feature vector is straightforward, we train a parametric model that has linear combination of feature vector with weight represent states. So from that view, we can easily construct an state representation lookup table by using an one-hot vector represent to state we want to talk about, and weight represent to the value of corresponding state.

We use GD to minimize the gap between our approximator and the real value function (`oracle`).

- But the big problem is, how we can take the ground-true value func (because it's not like supervised learning that we can directly assign one)? 

From the previous lecture, we know that we can substitute a target for $v\pi(S)$, using MC, TD and TD($\lambda$). By that, we can create an supervised training dataset to train agent.

- Why MC converges to local optimum while TD(0) converge (close) to global minimum?

When we get the an approximation value function, we just act greedy to the function and update it. It turns out not converge to optimal policy but just approximate optimal policy. This idea used for both State-value function approximation $v_{\pi}(S)$ and Action-value function approximation $Q_{\pi}(S,A)$

Using bootstap $\lambda$ usually help, the question is what hypyerparameter is good. 
In prediction, TD or TD($\lambda$) don't work in some situation. So gradient TD fixes that.

Q-learning using second network parameter to stably update the primary parameter. The dangerous of TD learning is everytime we update Q-value you also update Q-target.

Experience replay helps randomizes over the data, therefore help removing correlation in the observation sequences, or convert sequence date to i.i.d data. In the least square policy iteration, we use experienment replay to update parameter for stably updating. To use experience replay in batch method, we need to store transitions/ experiences $e _ { t } = \left( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } \right)$ in a `replay buffer`.

Hint: Interestingly, exprerience replay is the secret sauce that is [borrowed from Hassabis' area of the brain](https://www.technologyreview.com/s/532876/googles-intelligence-designer/) that helps RL works. “When you go to sleep your hippocampus replays the memory of the day back to your cortex.”

# Lecture 7: Policy Gradient Method 

In the lecture 6, we use an approximator to learn value functions ($V(S)$&$Q(S,A)$), from that we infer the policy. In this lecture, we will directly parameterize the policy by a distribution. Using value function approximation (value-based) perform poorly in partially observable enviroment or state aliasing (in that, different states that appear similar but require different responses so that we cannot infer just a deterministic policy).

**Optimising policy using object function J**: We define a objection function and base on some search methods (like GD, evolution search,...) to find $\theta$ that achieve optimal policy $\pi(\theta)$.

**To caculate the gradient of J**: finite differences or using Score function.

Score function take the form of how much unsual am i doing something. Using score function trick, we easily increase our objection func (reward) by just moving in the direction dertermined by the score_func*reward (where has positive reward, we move in the direction that get more that thing,or vice versa)

**Policy gradient** is a method that generalise likelihood ratio ($∇θJ(θ)$) approach to multi-step MDPs. $\nabla _ { \theta } J ( \theta ) = \mathbb { E } _ { \pi \theta } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( s , a ) Q ^ { \pi _ { \theta } } ( s , a ) \right]$

**Monte-Carlo Policy gradient** is a policy gradient method that sampling $$Q ^ { \pi _ { \theta } } ( s , a ) $$ by a return. This method behave varient and slow to converge.

**Actor-critic**: actor - the parametric approximator that pick action; critic - the parametric approximator that evaluate picked action. We can see critic as a leader and actor as a man follower
- **Reducing Variance Using a Baseline**: A Baseline $B(S)$ helps A-C reduce variance by substracting $B(s)$ from policy gradient, it doesn' change the direction of gradient, we can choose $B(s)=V_{\pi}(S)$ in the sense that "how much better than usual". $\nabla _ { \theta } J ( \theta ) = \mathbb { E } _ { \pi \theta } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( s , a ) A ^ { \pi _ { \theta } } ( s , a ) \right]$. It arises new task for critic which is estimate $V_{\pi}(S)$ to caculate Advantage func: $A ( s , a ) = Q _ { w } ( s , a ) - V _ { v } ( s )$
- This lecture also introduce **Natural Poclicy Gradient**, **Natural Policy Gradient**.

Summary: to estimate policy gradient, we take the score function multiply by value function/ return/ TD($\lambda$/ TD(0)/ baseline function, ...

# Lecture 8: Integrating Learning and Planning

This lecture cover model-based section, learning to model the world and plan in its MDP.

A model is parameterised by a tuple $< \mathrm { P } { \mathrm { I } } , \mathrm { R } { \eta } >$ - state of transition and reward. We infer the model just by counting or averaging the occurences ($<s_t+1,a_t+1>$) for all visit from each pair $<s_t,a_t>$. 

## Sample-based planning: 
A model is just used only for generating sampling (Simulated experience). After learning model, agent sampling S and R from its model and apply model-free RL method . (Learn about the world -> sample from our knowledge -> solve by model-free RL)>

## Dyna: 
Combine both learning from real experiences & planning from simulated experiences

Here is the Dyna algorithm:
![](https://raw.githubusercontent.com/thesunkid19/blog/gh-pages/img/dyna.png)
Dyna works better and more efficient than Q-learning.

## Simulation-Based search for planning
- Forward search: just look ahead and focus on the best action next stage and then next and next. We don't need to solve MDP, just sub-MDP from NOW (root node).
- Simulation-Based Search: Forward search using sample-based planning to simulate episodes of experiences $\left\{ s _ { t } ^ { k } , A _ { t } ^ { k } , R _ { t + 1 } ^ { k } , \ldots , S _ { T } ^ { k } \right\} _ { k = 1 } ^ { K } \sim \mathcal { M } _ { \nu }$, (capital character = sampled signal). Apply different model-free RL methods give diff search method: 
+ Simple Monte-Carlo Search: (1) give a model and a simulation policy $\pi$ (2) each $a \in \mathcal { A }$: Evaluate Q_value by mean return from simulated episodes (3) Select action in next state of root with max value.
+ Monte-Carlo Tree Search: Just Monte-Carlo control applied to simulated experience. Don't just look ahead, build a tree!
+ TD Search: Using TD instead of MC, that's it!


Summary: Planning, simulation-based search and effective method of planning it just use a model and apply to sample tranjectories to imagine what to happend next, build the tree and combine RL method (MC,TD,...)
**ζ**


# Lecture 9: Exploration and Exploitation 

Exploration & Exploitation is a dilemma in RL. This problem is about how to achieve longterm goal (get more information) but alse doesn't sacrifice short-term reward (just go ahead and take your coins)

There are few methods to approach this problem:
- Probability method: $\epsilon$-greedy, softmax, gaussian-noise
- Optimism in the Face of Uncertainty: just do which state/action with the highest uncertainty.
- Information state space: use a model to determine the helpful information in each state.

This lecture introduce new term: Regret. We calculate regret by $L _ { t } = \mathbb { E } \left[ \sum _ { \tau = 1 } ^ { t } V ^ { * } - Q \left( a _ { \tau } \right) \right]$ ($V ^ { * } = Q \left( a ^ { * } \right) = \max _ { a \in \mathcal { A } } Q ( a )$ which $V^{*}$ optimal value) - opportunity loss and find a way to minimize it, cause maximize(cumulative reward) <=> minimize(total regret). We try to find a lower bound for this function by the time-steps. Which is indicated by $\epsilon-greedy$, but $\epsilon-greedy$ seems not real because we don't have the gaps $\Delta _ { a } = V^* - Q(a)$. 

We can explore UCB and Thompson Sampling, it seems helpful but I don't fully understand yet.



 
  

