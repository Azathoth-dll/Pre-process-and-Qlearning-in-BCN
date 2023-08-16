
# Introduction

Boolean function , a kind of function that can be used to evaluate any Boolean
output associated with its Boolean input by means of a logical type of
calculation, plays a fundamental role in complexity theory. When Boolean
functions are applied to complex networks, the concept of a Boolean
Network (or BN for short) is defined: a Boolean Network is a set of
discrete Boolean variables, each assigned a Boolean function (which may
be different for each variable), that takes inputs from a subset of
these variables and outputs a state that determines the variable to
which it is assigned.

The Probabilistic Boolean Network was proposed by *Shmulevich
 et al. in 2002* and the model is an extension of the
Boolean network model . Each gene in this model corresponds to one or
more Boolean functions, and one of these functions is selected with some
probability at each iteration step to predict its next state.
Probabilistic Boolean networks retain some of the properties of Boolean
networks and can describe stochastic phenomena in gene regulatory
networks.

Reinforcement learning is a field of machine learning that emphasizes
how to act based on the environment in order to maximize the expected
benefit. This approach is generalizable and has therefore been studied
in other fields such as game theory, cybernetics, operations research,
information theory, simulation optimization, multi-subject system
learning, population intelligence, statistics and genetic algorithms.

With respect to probabilistic Boolean networks, many control theory
issues have been investigated for probabilistic Boolean control
networks, including observability, stabilization and output tracking,
which are mainly based on the semi-tensor product framework proposed by *Cheng et al*. However, most cannot be applied to gene regulatory
networks simulated by probabilistic Boolean network models. A Boolean
network with n nodes and a number of states of $2^n$ has tens of
billions of states for a Boolean model of a process with tens of
variables, while a model with thousands of variables has more states
than the Planck volume in the observable universe. The control problem
of Boolean networks, i. e. , making the system reach its target state by
control, has exponential time computational complexity and is even more
NP-hard (difficult to solve in polynomial time).

Although the control problem has been investigated using Q-Learning
and its variants, however, the results focus only on changing the level
of activation of specific genes and do not propose an explicit feedback
method to solve the control problem.

Compared to previous control problems with Boolean networks which have
studied small-scale networks, we use Q-learning algorithms in artificial
intelligence to study the control problem of large-scale Boolean
networks from an approximate perspective and solve its control and
NP-hard problem.

# PRELIMINARIES

In this section, we will declare notations and form the primary
structure of BCNs. Additionally, we adopt one of the Reinforcement
learning methods, Q-Learning, on the control problem of BCNs. Moreover,
we introduce a method of pre-process to reduce calculation.

## Markov Decision Process

A Markov Decision Process(MDP) contains two sections, the environment
and the agents. In MDP, each agent has target states. The agent updates
its state by interacting with the environment and receives rewards by
taking actions. The process continues until the agent reaches a target
state, which marks the end of an episode. Meanwhile, the process of each
state change by the agent is called a time step. The framework of an MDP
at time step $t$ is provided in Fig. 1.

For Agent $i$ ,

![Framework of MDP](raw.githubusercontent.com/Azathoth-dll/Pre-process-and-Qlearning-in-BCN/master/img/fig_rl.png)

## Digraph

A digraph is defined as a set of vertices and a collection of directed
edges, each connecting an ordered pair of vertices. We say that a
directed edge points from the first vertex in the pair and points to the
second vertex in the pair.

A directed path in a digraph is a sequence of vertices in which there is
a (directed) edge pointing from each vertex in the sequence to its
successor in the sequence, with no repeated edges. 

**definition 1**. *If there is a directed path from vertex $A$ to vertex
$B$ and a directed path from vertex $B$ to vertex $A$, A and B are
strongly connected.*


## Boolean Control Network

### Modelling with formula

We denote the state vector of the BCN at timestep $t$ by
$x(t)=(x_t^1,\dots,x_t^n)$, action vector of BCN at timestep $t$ by
$\mathcal{U}(t) = (u_t^1,\dots,u_t^n)$, where $\mathcal{U}(t)$ is in
$\mathcal{U} = \{\mathcal{U}(t),t\in \mathbb{Z} ^+\}, \mathcal{U}\subseteq \mathcal{B}^m$.
Similarly, we denote the target state by $x_d=(x^1,\dots,x^n)$ and the
optimal policy by $\pi^\ast =(u^1,\dots,u^m)$.
$\mathcal{U}(t) \in\mathcal{B} ^m$ where $\mathcal{B}$ is defined as
$\{ 0,1\}$, so $u_t^i\in\mathcal{B}$, and $\mathcal{B}^m$ is
$\underbrace{\mathcal{B}\times \dots\times\mathcal{B}}_{m}$. Similarly,
we get $x_t=(x_t^1,\dots,x_t^n)\in\mathcal{B}^n$\
A BN with $n$ nodes can be expressed at timestep $t+1$ as:
$$\begin{cases}
        x_{t+1}^1 = f_1(x_t^1,\dots,x_t^n), \\
        \vdots                                     \\
        x_{t+1} = f_n(x_t^1,\dots,x_t^n). 
    \end{cases}$$
where $x_j^i$, whose value selects only between 0 or 1 , denotes the
values of $node_i$ at timestep $j$. $f_i$ is the iteration operator, or
can be named as state transition function, for $node_i$ without impact
of control nodes , donating Boolean operation on nodes at timestep
$i-1$.

We employ $g$ to denote the control operator so that we get the full
expression of BCN:
$$\begin{cases}
        x_{t+1}^1=g\cdot f_1(x_t^1,\dots,x_t^n), \\
        \vdots                                             \\
        x_{t+1}^n=g\cdot f_n(x_t^n,\dots,x_t^n). \\
    \end{cases}$$
Simply, we denote as $x_{t+1} = g\cdot f(x_t)$. It worths mentioning
that $g$ may reverse the values of some of the nodes in BN or apply
control nodes,whose value rely on $u_t^i,i=(1,\dots,m)$ at timestep $t$,
on the BCN.\
We have $\wedge,\vee,\neg,\oplus$ represent the basic Boolean operator
AND, OR, NOT, XOR respectively.

### Modelling with graph

We denote $\mathcal{X}$ the states space,
$\mathcal{X} = \{x_t,t\in\mathbb{Z}^+\}\subseteq \mathcal{B}^n$ which
contains all the states the agent may achieve. If the agent switches
from state $x_t$ to state $x_{t+1}$ through state transition
function$f$, we can define that there is an edge between the vertex
$x_t$ and vertex $x_{t+1}$ whose direction is from $x_t$ to $x_{t+1}$.
We employ a set $E(\mathcal{X})$ to store all the directed edge of
$\mathcal{X}$ defined like this. As a result we have a digraph whose
vertices set is $\mathcal{X}$ and edges set is $E(\mathcal{X})$.

 Rem
**remark 1**. *State $x_i$ can turn into state $x_j$ if and only if
there is a path from vertex $x_i$ to vertex $x_j$ in the corresponding
digraph.*


This is obvious because if there's a path from vertex $x_i$ to vertex
$x_j$, there must be a sequence of vertices
$\{x_i,x_{i1},\dots,x_{ik},x_j\}$ and a set of directed edges connecting
from $x_i$ to $x_j$. According to the graph model, the agent can
switches from $x_i$ to $x_{i1}$, from $x_{i1}$ to $x_{i2}$, and so on.
So it can switches from $x_i$ to $x_j$.

It should be pointed out that: The content of the graph is partially
invisible to us mainly because the environment of the MDP is unknown. In
other words, we can not figure out the state transition function $f$.
However, we can still study some of the properties of the graph, which
can help us optimize our algorithm.

## Reinforcement Learning

Reinforcement Learning works under the framework of MDP, which can be
described as $(\mathcal{X},\mathcal{U},P,\gamma,r)$. As is mentioned
above, $\mathcal{X}$ is the states space and
$\mathcal{X}\subseteq \mathcal{B}^n$,
$\mathcal{X}=\{x_t,t\in\mathbb{Z}  ^+\}$; $\mathcal{U}$ is the action
space, $\mathcal{U} = \{\mathcal{U}(t),t\in \mathbb{Z} ^+\}$; P is the
set of state transition probability; $\gamma$ is the discount factor;
$r_t = r(x_t) = \mathbb{E}[r(x_t,u_t)| x_t,u_t,x_{t+1}]$, where
$\mathbb{E}(\cdot )$ is the expectation operator.\
As mentioned earlier, $x_d$ is the set of target states and
$\pi^\ast =(u^1,\dots,u^m)$ is the set of optimal policy.\
The reward function $r(x(t))$ is defined as below:
$$r(x_t,u_t) = \begin{cases}
1,x_{t+1} \in x_d\\
0,x_{t+1} \notin x_d
\end{cases}$$

Let's think about the process of rewards payback carefully. When the
agent whose state is $x_t$ updates its state, it gets the reward
$r(x_t,u_t)$ from the environment, and at timestep $t+1$, it gets the
reward $r(x_{t+1},u_{t+1})$ and changes its state into $x_{t+2}$.
$\gamma$ is the discount and $G(t)$ is the cumulative reward of the
original state $x_t$. Then the cumulative reward from timestep $t$ to
$T$ is $\gamma r(x_t,u_t) + r(x_{t+1},x_{t+1})$. We employ $r_t$ as the
actual reward in training when the agent chooses any action and shifts
its state. The cumulative reward from timestep $t$ to $T$ is
$G(t)=\sum_{i=t}^T \gamma^{i-t}r_i$.\
According the description of MDP above, the process can be described as
$S-A-R-S-A$, which is similar to BCNs. Additionally, in the process of
RL training, the agent remembers the reward got at state $x_t$ and tries
to rearrange its policy in the next Episode. At the following paragraph,
we introduce state-value function and action-value function to store the
reward. We employ the state-value function by
$$v_\pi(x_t) = \mathbb{E}_\pi[G(t)|x_t],$$ 
similarly
$$q_\pi(x_t,u_t) = \mathbb{E}_\pi[G(t)|x_t,u_t],$$
 where
$\pi = \{u_t|t\in \mathbb{Z}^+\}$ is the policy, $\pi\in\Pi$ which is
the set of policy.

From Bellman equation, we describe the iteration of the value functions:

$$v_\pi(x_t) \!= \!\sum_{x_{t+1}\in\! \mathcal{X}}\!P_{x_t}^{x_{t+1}}(\pi(x_t))[r_\pi(x_t)\!+\!\gamma v_\pi(x_{t+1})]$$
$$q_\pi(x_t,u_t) \!= \!\!\sum_{x_{t+1}\!\in \mathcal{X}}\!\!P_{x_t}^{x_{t+1}}\!(\!u_t\!)\![r(\!x_t,u_t\!)\!+\!\gamma q_\pi(\!x_{t+1},\!u_{t+1}\!\!)]$$
so we can express the optimal value functions in:
$$v^\ast (x_t)=\max _{\pi\in\Pi}v_\pi(x_t),\forall x_t\in \mathcal{X}$$
$$q^\ast (x_t,u_t)=\max _{\pi\in\Pi}q_\pi(x_t,u_t),\forall x_t\in \mathcal{X},\forall u_t\in \mathcal{U}$$

## Q-Learning

As is mentioned, state transition probability $P$ is unknown, so it's
not quite practical to use the formulas above in training because they
require $P$. From the other hand, the iteration of value function
$v_\pi(x_t)$ and $q_\pi(x_t)$ require the value functions at timestep
$t+1$, which rely on the agent choosing the next action. In fact, we
choose Q- Learning which select a direct approximation for optimal
action value function $q^\ast$.\
The update of the action value function $q_\pi(x_t,u_t)$ Q-Learning(QL)
can be described as:
$$q_\pi\!(\!x_t\!,\!u_t\!)\! \leftarrow \!q_\pi\!(\!x_t\!,\!u_t\!)\! +\! \alpha\![r_{t+1}\!+\!\gamma\!\max _u \!q_\pi\! (\!x_{t+1}\!,\!u\!)\!-\!q_\pi(\!x_t,\!u_t\!)\!]$$
where $a$ is the learning rate and $a\in(0,1]$.

# BCN using Q-Learning with Pre-process

## Optimize QL in BCN with Pre-process

In a BCN, we use QL to find out the optimal control policy $\pi^\ast$
which should satisfy:
$$\pi^\ast = \max_{\pi\in\Pi}\mathbb{E}_\pi [G(t)],\forall t \in \mathbb{Z}^+,$$
at the same time, the control nodes should as few as possible.

When we model a BCN with a digraph, the control problem of reachability
can be transferred into finding a path to the target state. In other
words, if it doesn't exist, the BCN can't reach the target state.

**definition 2**. *For a BCN modeled by graph, state $x_i$ and $x_j$
have corresponding vertices $x_i$ and $x_j$ respectively in graph. If
vertex $x_i$ and $x_j$ are strongly connected, state $x_i$ and $x_j$ is
called equivalent.*


Obviously, if the BCN can't be controlled, there must be at least two
sets of equivalent vertices. As a result, the problem of adding control
nodes is turned into establishing a path between the set of vertices
that contains the vertex, which represents the target state, and the
other sets. We name the equivalent points set as equivalent points sets.
Especially, we name the equivalent points set which contains the target
state as the target set.

## Pre-process and Q-Learning

We denote $D_{x_t}^{x_d}$ the Manhattan Distance to measure the the
difference between the state of the agent at timestep $t$. Because of
the characteristic of boolean network, $D_{x_t}^{x_d}$ is equivalent to
the number of nodes in BCN which don't reach target state.

**definition 3**. *$A$ and $B$ are two equivalent points sets. Define
the distance between $A$ and $B$:
$$D_A^B = \min_{x,y} D_x^y, x\in A , y \in B,$$


Given the two state vectors $x$ and $y$ which satisfy Definition 2, we
can find out $x^i$ which satisfies
$x^i \neq y^i;i=1,\dots,n;x^i \in x;y^i \in y$. Obviously, $x^i$
correspond to node $i$ in the BCN. If we add control node on this node
to change its value, $x$ will turn into $y$. As a result, we get a basic
method to find out the add control nodes to BCN. However, there still
lots of control nodes can be added under the circumstance because the
two-tuples which satisfy Definition 2 may not be unique as well as the
average speed of reaching target state may be different. So we adopt RL
method, especially QL, to work out the optimal control policy.

We employ
$Q_t = \{q_\pi(x_t,u_t),(x_t,u_t)\in\mathcal{X}\times \mathcal{U}$\}, the
Q-table at timestep $t$, in which the action value is stored.

**theorem 1**. *For a BCN with $n$ nodes, state $x_0$ can be controlled
to reach the target state if and only if
$\exists T,s. t. \exists u,Q_T(x_0,u)>0$.*

*Proof.* (Sufficiency)\
Under the circumstance that there exists $T$ satisfying
$\exists u,Q_T(x_0,u)>0$. Without loss of generality, we suppose there
is only one $x_d$ and one $x_0$. According to the definition of
$Q_T(x_0,u)$ and the Bellman equation, we have
$Q_T(x_0,u) = q_\pi(x_0,u) = \mathbb{E}_\pi[G(T)|x_0,u]$, so that if
$\exists T,s. t. \exists u,Q_T(x_0,u)>0$, we can conclude that
$\exists G(T)>0$, so that $\exists t,s. t. r(x_t,u_t)>0$, which means
that $x_t$ will turn into $x_{t+1}$ under action $u_t$ where
$x_{t+1} = x_d$. As a result, the BCN at state $x_0$ can be controlled
to the target state.\
(Necessity)\
Without loss of generality, we suppose there is only one $x_d$ and one
$x_0$.\
If state $x_0$ can be controlled to reach the target state,according to
the definition of $r(x_T,u_T)$, it must establish that $r(x_T,u_T) >0$
where $x_{T+1} = x_d$. As a result,
$r_T = \mathbb{E}[r(x_T,u_T)| x_T,u_T,x_{T+1}] > 0$, so that
$G(0)=\sum_{i=0}^T \gamma^i r_i > \gamma^T r_T >0$

Considering the contradictory situation, if
$\forall T,s. t. \forall u,Q_T(x_0,u)=0$, we can conclude similarly from
Sufficiency that
$Q_t(x_0,u) = q_\pi(x_0,u) = \mathbb{E}_\pi[G(0)|x_0,u]  = 0$ because of
$G(0) = 0$, which cause a contradiction. ◻

---
#### Algorithm 1
```python
X_equ = X_d.copy()
    for x_d in X_d:
        x_t = x_d
        t = 0
        while t < N:
            x_tt = system(x_t)
            if x_tt != x_d:
                X_equ.append(x_tt)
            else:
                break
            t = t + 1
            x_t = x_tt
```
---
We try to find out the optimal control policy. If we use enumeration
method to find out as the Training set of QL, it may take a lot of time
facing to a large-scale BCN with more than 9 nodes. So we tend to
eliminate the initial control nodes in training set. As is mentioned
before, we should find out the Manhattan Distance between the ordinary
equivalent points sets and target sets. Additionally, we should work out
the pairs of points which satisfy the distance. And then we will figure
out which components of the corresponding state vectors cause the
increase of distance, which correspond to the node in BCN to be added
control nodes.\
The [Algorithm 1](#algorithm-1) is to find out the target sets. Specially, we assume
that different target sets own equal importance, so that we can combine
them in one target set.

---
#### Algorithm 2

```python
while t < N:
        i = 0
        while(1):
            k = random.randint(0,2 ** n - 1)
            x_0 = X_0[k]
            if not(x_0 in X_equ):
                x_t = x_0
                break

        while i < tau:
            x_t1 = system(x_t)
            for y in X_equ:
                if D(x_t,y) < D_0:
                    D_0 = D(x_t)
                    M = M.append()
            i = i + 1
            if x_t1 == x_0:
                break
        x_t = x_t1
        t = t + 1
    for a in M:
        Mc = 0
        for i in range(0,n):
            if a[0] & (2 ** i) != a[1] & (2 ** i):
                Mc = Mc + 2 ** i
        C.append(Mc)
        return C
```
---

We tend to find out the pairs of points mentioned before and store them
in a tuple $M$. After that, we are going to find out the components
which cause the difference and store the information in terms of index
in the tuple $Mc$. Obviously, we won't explore each states because there
is nearly no difference with enumeration method. So we add limitation to
the number of states to be considered.

Now that we have $C$ which is the set of all possible control nodes. As
is mentioned before, we are going to optimize the control policy with
QL. Of course, for the limitation we placed in [Algorithm 2](#algorithm-2), there may be
some states which can't be controlled to target states even if we placed
control nodes according to $C$. So in Algorithm 3, when it comes to the
states like this, we tend to back to [Algorithm 2](#algorithm-2). Of course, this will
raise up some problems to be discussed later.

---
#### Algorithm 3
```python
def control(X_0,X_d,C,N,gamma,omega,n,T):
    # For fig
    if(Switch):
        C_now = []
        X = []
        for i in range(1 << n):
            C_now.append(0)
        Err = []
        Loss = []

    # For train
    Qtable = {}
    epsilon = 1
    # C_power = powerset(C)
    # start to train
    for episode in range(0,N + 1):
        
        alpha = 1 / ((episode + 1) ** omega)
        i = random.randint(0,len(X_0))
        x_t = X_0[i-1]
        t = 0
        
        while (t < T):

            if (Switch):
                i_for_fig = C.index(x_t)
                X.append(x_t)

            if not (x_t in Qtable):
                Qtable[x_t] = {}
                for c in C:
                    Qtable[x_t][c] = 0
            rndm = random.random()
            if rndm > epsilon:
                Q_umax = 0
                for u in Qtable[x_t]:
                    if Q_umax < Qtable[x_t][u]:
                        Q_umax = Qtable[x_t][u]
                        u_max = u
            else:
                u_max = random.choice(C)
            if epsilon > 0.01:
                epsilon = epsilon - 0.99 / N

            if (Switch):
                C_now[i_for_fig] = u_max

            x_tt = system(x_t) ^ u_max


            if not(x_tt in Qtable):
                        Qtable[x_tt] = {}
                        for c in C:
                            Qtable[x_tt][c] = 0

            Q_tmax = 0
            for u in Qtable[x_tt]:
                if Q_tmax < Qtable[x_tt][u]:
                    Q_tmax = Qtable[x_tt][u]

            q_t = Qtable[x_t][u_max] + alpha * (retn(x_tt,X_d) + gamma * Q_tmax - Qtable[x_t][u_max])

            Qtable[x_t][u_max] = q_t

            if x_tt in X_d:
                break
            else:
                x_t = x_tt
                t = t + 1
        
        
        d = 0

        if (Switch):
            for i in range(len(C_now)):
                d = d + D(C_now[i],mu[i],n)
            Loss.append(d)
            if len(Loss) >= K:
                err = 0
                for value in Loss:
                    err = err + value
                Err.append(err / K)
                Loss = []

    if (Switch):
        Y = np.array(Err)
        plt.xlabel('Episode')
        plt.ylabel('Average Error')
        plt.plot(Y,'o-b',markersize = 2)
        plt.show()


    return Qtable

def optpolicy(Qt):
    qt = {}
    for x_t in Qt:
        Qu_max = 0
        for u in Qt[x_t]:
            if Qt[x_t][u] > Qu_max:
                Qu_max = Qt[x_t][u]
                u_max = u
        qt[x_t] = u_max
    mu = []
    for i in range(1 << n):
        mu.append(qt[i])
    if(not Switch):
        print(mu)
    
    return mu
```
---

# Simulation

In this section, we will show our algorithm with two examples. They are
a small-scale system with ***3*** nodes and a larger-scale system with ***9*** nodes.

Computations were run using a *__10-Core Intel i7-12650H__* processor, ***2. 3
GHz*** frequency, ***32 GB RAM* and *Python*** software. In both examples, we
set $\gamma = 0. 9$. What's more, we set
$\alpha = 1 / ((ep + 1) ^ \omega)$ where episode donating episode and
$\omega = 0. 75$. We adopt the $\epsilon$-greedy policy with a linear
decaying value from 1 to 0.01.\
It's worth mentioning that we adopt bitwise calculation rather than
python list operation when processing and storing statistics in order to
decrease memory and speed up operations. In other words, we use a
number's binary form in computer to store the corresponding states and
actions.

To evaluate the performance of the algorithm, we define the K episodes
average error:

#### average error
$$e(k) = \frac{1}{K}\sum_{t = k}^{k + K -1}L_t, k=0,K,\dots,N-K$$ 
    where $L_t$ represents the loss function,
#### loss function
$$L_t = \|C - \mu_t(\cdot)\|$$
where C is the optimal control policy and
$\mu_t(\cdot)$ is the optimal control policy.

## Small-scale system with three nodes

We model a small-scale system with three nodes as follow:
$$\begin{cases}
    x_{t+1}(0) = x_t(1) \wedge x_t(2)\\
    x_{t+1}(1) = \neg x_t(0) \vee x_t(2)\\
    x_{t+1}(2) = \neg x_t(1)
\end{cases}$$ 
We set the set of target state, $X_d = \{1\}$, and the set
of initial state $X_0 = \mathcal{X}$, where $\mathcal{X}$ is the set of
all states of agent. What's more, we set $N = 10^5$ and $K = 5000$.
What's more, $T > 2^n$ and we choose T = 20\
By using Algorithm 3, we receive the optimal controllers:
$$\begin{cases}
    x_{t+1}(0) = x_t(1) \wedge x_t(2) \oplus u^1 \\
    x_{t+1}(1) = \neg x_t(0) \vee x_t(2) \oplus u^2\\
    x_{t+1}(2) = \neg x_t(1) \oplus u^3
\end{cases}$$

Through formula [error](#average-error) and [loss](#loss-function), we acquire the average error in $K$ episode, which is
shown in following figure:

![Average Error in K episodes of given 3-nodes system](raw.githubusercontent.com/Azathoth-dll/Pre-process-and-Qlearning-in-BCN/master/img/Fig1.png)

In this case, it takes 1.11 *sec* to finish the training. It is obvious
from the figure that the optimal policy is not unique. And we get one of
the optimal policies is $\{5,7,1,4,5,7,4,6\}$.

## Larger-scale system with nine nodes

The system is modeled as below: $$\begin{cases}
    x_{t+1}(0) = x_t(1) \wedge x_t(2) \wedge x_t(7)\\
    x_{t+1}(1) = \neg x_t(0) \vee x_t(2) \wedge x_t(8)\\
    x_{t+1}(2) = \neg x_t(1)\\
    x_{t+1}(3) = \neg x_t(0) \wedge x_t(4) \wedge x_t(5) \vee \neg x_t(6)\\
    x_{t+1}(4) = x_t(2) \vee x_(t)_3\\
    x_{t+1}(5) = \neg x_t(7)\\
    x_{t+1}(6) = x_t(0) \wedge x_t(2)\\
    x_{t+1}(7) = x_t(2)\\
    x_{t+1}(8) = x_t(1) \wedge \neg x_t(5)\\
\end{cases}$$ In this example, the target states we set is
$X_d = \{1,8,234,124,500\}$. For the system with 9 nodes, there exist
$2 ^ 9$ states, which means we need more episodes to train out the
optimal policy. So we choose $N = 10^5, K = 5000$, and
$T > |\mathcal{B}^9|$ which is set 600. And Similarly, we evaluate the
algorithm through formula [error](#average-error) and [loss](#loss-function) and give out the following figure:

![Average error in K episodes of given 9-nodes system](raw.githubusercontent.com/Azathoth-dll/Pre-process-and-Qlearning-in-BCN/master/img/Fig2.png)

Because of the number of states and multiple target states, the error
function can't converge.
