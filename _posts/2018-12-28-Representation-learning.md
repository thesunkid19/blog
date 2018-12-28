---
layout: post
title: "Quick note about representation learning"
date: 2018-12-28
categories: blog
tags: [DL,ML]
---
## Deep learning from representation learning perspective
----

`“Sometimes it happens that a man’s circle of horizon becomes smaller and smaller, and as the radius approaches zero it concentrates on one point. And then that becomes his point of view.” — David Hilbert`

![](https://raw.githubusercontent.com/thesunkid19/blog/gh-pages/img/representation-ahung.jpg)

Every task in machine learning/ deep learning can be acknowledged from point of the view of representation learning. (thought distribution learning) 
- In CNN: we find a function in function space that map input from sensor (image) to latent spartial vector which represent our thought about these context & object in the image. We use some deep prior to  make some constraint that help optimizing/finding process is easier (if not, we need more large data and computational resources), it's spartial pattern, local feature and sharing feature capture (express in filters)
- In RNN: we find a function (that people usually use the term `weight-sharing` model - Recurrent network) that take input from latent context vector from time step t-1 (summary lossy of the history) and input of current state to make new latent context vector. It is also representation state in Reinforcement learning perspective. 
- In reinforcement learning, agents make decision by manipulating their latent space operands, that transform information from context vector to decision-planning vector in their thought (what is this?). It not only use knowledge from context vector but also integrate some module knowledge from modular high-level knowledge that relavant (near in cluster modular concept space) to specific task, which are these thing that agent had learned and distilled in the past.
- In above picture, a Hung said that concept is the central point of though distribution. When we are receptive to new input, our model transform input to a distribution and when we need to generative an output from thought to describe this though with the world, we sample from that distribution (ex: when we said "Hoa", have a distribution about many `things` name "Hoa" in our head.)
 

**Additional resources:**
1. [Yoshua Bengio talk on Deep learning](https://www.youtube.com/watch?v=azOmzumh0vQ), some points about representation learning:
- Learn disentangled representation
- Catastrophic forgetting problem 
2. [Integrating State Representation Learning into Deep Reinforcement Learning](http://www.jenskober.de/publications/deBruin2018RA-L.pdf) 
3. [An overview of representation learning 2014](https://arxiv.org/pdf/1206.5538.pdf)
4. [From Deep Learning of Disentangled Representations to Higher-level Cognition - Yoshua Bengio 2018] (https://www.youtube.com/watch?v=Yr1mOzC93xs)
5. [State Representation Learning for Reinforcement Learning](https://www.youtube.com/watch?v=mx6L-QJMYqQ)
6. [Adaptive Representations for Reinforcement Learning - Shimon Whiteson](http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/whitesonbook10.pdf) 
7. [Chapter 15: Representation learning - Deep learning book](https://www.deeplearningbook.org/contents/representation.html)
	



 


