## Control System

Although control theory has been firmly rooted in model-based design philosophy, abundant data and machine learning methods have brought new opportunities to control theory. The intersection of control theory and machine learning encompasses a wide range of research directions and applications in various real-world systems.

### Linear Quadratic Control

On the theoretical side, Linear-Quadratic Control is a classical control method. If a dynamical system can be represented by a set of linear differential equations and its constraints are quadratic functionals, such problems are called linear-quadratic problems. The solution to such problems is the Linear-Quadratic Regulator (LQR). Recently, there has been research on graph neural networks in distributed linear-quadratic control, which transforms linear-quadratic problems into self-supervised learning problems, enabling the discovery of optimal distributed controllers based on graph neural networks. The researchers also derived sufficient conditions for the stability of the resulting closed-loop systems.

### Model Predictive Control

Model Predictive Control (MPC) is an advanced process control method used to control a process while satisfying a set of constraints. The main advantage of MPC is that it allows optimization of the current time step while considering future time steps. Therefore, it differs from the Linear-Quadratic Regulator. MPC also has the ability to predict future events and can take control actions accordingly. Recent research has combined optimal control and machine learning and applied them to visual navigation tasks in unfamiliar environments: for example, a learning-based perception module generates a series of waypoints to guide the robot to the target through collision-free paths, a model-based planner uses these waypoints to generate smooth and dynamically feasible trajectories, and then feedback control executes them on the physical system. Experiments show that compared to purely geometry-based mapping or end-to-end learning-based approaches, this new system can reach the target position more reliably and efficiently.

### Stability Analysis of Control Systems

Because safety is critical for robotic applications, some reinforcement learning methods improve safety by learning the uncertainties of dynamics, encouraging methods that are safe, robust, and can formally certify learned control policies. :numref:`safe\_learning\_control` shows the framework diagram of a Safe Learning Control system. Lyapunov functions are effective tools for evaluating the stability of nonlinear dynamical systems, and recently Neural Lyapunov has been proposed to incorporate safety considerations.

![Safe Learning Control System, where data is used to update the control policy or safety filter :cite:`brunke2021safe`](../img/ch13/safe_learning_control.png)

:width:`800px`

:label:`safe\_learning\_control`

