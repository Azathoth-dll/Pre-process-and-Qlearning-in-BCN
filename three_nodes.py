import random
import numpy as np
import matplotlib.pyplot as plt
import time
#### small scale system with 3 nodes



#_______________________###### PUBLIC ######___________________________#
### Manhattan Distance
# D(x,y) = x ^ y
def D(x,y,n):
    orx = x ^ y
    d = 0
    for i in range(0,n):
        if orx & (1<<i):
            d = d + 1

    return d

### return
def retn(x_tt,X_d):
    if x_tt in X_d:
        rt = 1
    else:
        rt = 0
    return rt

### Control
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

#_______________________###### LOCAL ######___________________________#
### system
# x_0 = x_1 & x_2
# x_1 = ~ x_0 | x_2
# x_2 = ~ x_1
# x_0 = (x_d & 1)
# x_1 = (x_d & 2)
# x_2 = (x_d & 4)
def system(x_t:int)->int:
    x_tt = (((x_t & 2) >> 1) & ((x_t & 4) >> 2)) | (((~(x_t & 1)+2)|((x_t & 4)>>1))<<1) | ((~((x_t & 2)>>1)+2)<<2)
    return x_tt


### time
timeStart = time.time()

### target states
# x_d <= 2 ^ 3 = 8
# X_d =[1] 

X_d = [3]
X_0 = [0,1,2,3,4,5,6,7]


### Initial Control Nodes
# Input:
# X_0, X_d, K, tau, n
# Output:
# C
n = 3
N = 100000
T = 100
gamma = 0.9
omega = 0.75

Switch = 0

### Debug
# C = iniControl(X_d,X_0,N,tau,n)
K = 5000
C = X_0

# mu = [5, 7, 1, 3, 5, 7, 4, 6]
Qt = control(X_0,X_d,C,N,gamma,omega,n,T)
timeend = time.time()

print(timeend - timeStart)
mu = optpolicy(Qt)
### optimal control