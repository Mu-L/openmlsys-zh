## Planning System

Robot planning includes not only motion path planning but also task planning :cite:`9712373` :cite:`wang2023mimicplay`, :cite:`li2023behavior`. Among these, motion planning is one of the core problems in robotics, finding a path for the robot between two given positions that satisfies constraint conditions. These constraints can include collision-free paths, shortest paths, minimum mechanical work, etc., and require guarantees of probabilistic completeness and optimality. Motion planning has applications ranging from navigation to robotic arm manipulation in complex environments. However, challenges remain when classical motion planning deals with real-world robotic problems in high-dimensional spaces. Researchers are still developing new algorithms to overcome these limitations, including optimizing computation and memory load, improving planning representations, and handling the curse of dimensionality.

At the same time, some advances in machine learning have opened new perspectives for robotics researchers studying motion planning problems: addressing the bottlenecks of classical motion planners in a data-driven manner. Deep learning-based planners can perform planning using visual or semantic inputs. ML4KP is a C++ library for motion planning with kinodynamic systems that can easily integrate machine learning methods into the planning process.


Reinforcement learning also has important applications in planning systems :cite:`sun2021adversarial`. Recently, some works have been based on the MetaDrive simulator :cite:`li2021metadrive` for multi-agent reinforcement learning, driving behavior analysis, etc. :cite:`peng2021learning` :cite:`peng2021safe` :cite:`li2021efficient`. To better illustrate how reinforcement learning is applied in autonomous driving, especially as a planning module for autonomous driving, :numref:`rl\_ad` shows a deep reinforcement learning-based autonomous driving POMDP model, including important components such as the environment, reward, and agent.

![Deep Reinforcement Learning-Based Autonomous Driving POMDP Model :cite:`aradi2020survey`](../img/ch13/rl_ad.png)

:width:`800px`

:label:`rl\_ad`
