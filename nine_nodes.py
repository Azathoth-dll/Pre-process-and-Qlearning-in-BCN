import random
import numpy as np
import matplotlib.pyplot as plt
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
        Y=np.array(Err)
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


#_______________________________LOCAL__________________________________#
### system
# x_0 = x_1 & x_2 & x_7
# x_1 = ~ x_0 | x_2 & x_8
# x_2 = ~ x_1
# x_3 = ~ x_0 & x_4 & x_5 | ~ x_6
# x_4 = x_2 | x_3
# x_5 = ~ x_7
# x_6 = x_0 & x_2
# x_7 = x_2
# x_8 = x_1 & ~ x_5


def system(x_t:int)->int:
    x_tt = (x_t >> (1 << 1) & 1) & (x_t >> (1 << 2) & 1) & (x_t >> (1 << 7) & 1) | ((~ (x_t >> (1 << 0) & 1) + 2) | ((x_t >> (1 << 2) & 1) & (x_t >> (1 << 8) & 1))) << 1 | (~ (x_t >> (1 << 1) & 1) + 2) << 2 | ((~ (x_t >> (1 << 0) & 1) + 2) & (x_t >> (1 << 4) & 1) & (x_t >> (1 << 5) & 1) | (~ (x_t >> (1 << 6) & 1) + 2)) << 3 | ((x_t >> (1 << 2) & 1) | (x_t >> (1 << 3) & 1)) << 4 | (~ (x_t >> (1 << 7) & 1) + 2) << 5 | ((x_t >> (1 << 0) & 1) & (x_t >> (1 << 2) & 1)) << 6 | (x_t >> (1 << 2) & 1) << 7 | ((x_t >> (1 << 1) & 1) & (~ (x_t >> (1 << 5) & 1) + 2)) << 8 
    return x_tt


### target states
# x_d <= 2 ^ 3 = 8
# X_d =[1] 
X_d = [1,8,234,124,500]

X_0 = [32,43,15,0,54,442,123,415]

C = []
for i in range(512):
    C.append(i)


### Initial Control Nodes
# Input:
# X_0, X_d, K, n
# Output:
# C
n = 9
N = 1000000
T = 60
gamma = 0.9
omega = 0.75
K = 50000
Switch = 1


### Debug
mu = [272, 207, 35, 472, 491, 299, 449, 297, 182, 82, 472, 45, 290, 290, 339, 220, 160, 330, 61, 284, 483, 454, 274, 274, 231, 330, 165, 22, 434, 336, 417, 186, 48, 85, 198, 45, 371, 290, 450, 196, 132, 474, 251, 198, 26, 337, 297, 24, 28, 127, 459, 253, 454, 336, 388, 274, 191, 182, 253, 22, 78, 138, 505, 1, 82, 138, 198, 36, 509, 448, 288, 489, 239, 196, 134, 80, 509, 222, 511, 449, 160, 84, 21, 135, 443, 454, 388, 274, 194, 182, 244, 43, 336, 434, 361, 462, 85, 85, 117, 80, 371, 448, 489, 288, 348, 119, 87, 117, 290, 26, 491, 288, 182, 191, 264, 253, 434, 443, 200, 388, 182, 127, 253, 22, 388, 443, 496, 12, 119, 38, 80, 45, 222, 290, 340, 441, 82, 38, 80, 80, 47, 337, 220, 220, 182, 191, 460, 226, 379, 443, 505, 496, 194, 182, 264, 253, 78, 336, 24, 387, 47, 47, 198, 284, 222, 299, 409, 491, 38, 47, 117, 197, 398, 411, 220, 310, 330, 84, 161, 264, 483, 434, 388, 505, 398, 330, 253, 22, 336, 132, 417, 12, 196, 38, 251, 198, 290, 118, 369, 22, 474, 85, 45, 45, 451, 342, 297, 22, 197, 105, 244, 128, 483, 90, 207, 388, 398, 84, 165, 86, 454, 483, 12, 417, 47, 62, 466, 80, 448, 290, 339, 297, 82, 47, 22, 368, 337, 342, 340, 310, 160, 127, 22, 244, 336, 437, 452, 484, 327, 194, 165, 165, 365, 422, 496, 387, 99, 63, 34, 434, 292, 306, 311, 294, 233, 458, 101, 476, 292, 326, 466, 466, 1, 191, 244, 22, 377, 377, 505, 12, 84, 384, 22, 253, 90, 336, 198, 496, 94, 63, 456, 52, 306, 315, 323, 8, 63, 66, 64, 258, 315, 63, 495, 324, 194, 350, 22, 128, 434, 454, 12, 505, 191, 125, 165, 264, 454, 336, 505, 303, 315, 49, 456, 253, 327, 326, 353, 294, 212, 54, 453, 456, 4, 507, 204, 304, 182, 191, 135, 22, 420, 434, 505, 12, 227, 84, 284, 77, 454, 483, 387, 486, 212, 99, 214, 456, 206, 507, 466, 353, 215, 255, 61, 52, 306, 493, 204, 466, 398, 398, 43, 61, 443, 454, 303, 12, 105, 330, 239, 450, 443, 336, 313, 452, 103, 66, 64, 235, 315, 321, 304, 304, 233, 458, 456, 101, 206, 102, 353, 466, 231, 182, 226, 253, 483, 379, 0, 417, 84, 231, 244, 253, 78, 78, 417, 388, 27, 448, 64, 61, 10, 464, 357, 313, 66, 99, 61, 158, 306, 326, 466, 324, 47, 191, 244, 128, 443, 365, 303, 387, 231, 28, 135, 272, 78, 415, 496, 486, 454, 212, 64, 34, 306, 321, 353, 323, 212, 54, 456, 101, 326, 315, 106, 304, 84, 84, 460, 244, 365, 336, 303, 417, 191, 182, 253, 244, 504, 336, 417, 387, 458, 255, 268, 61, 326, 355, 353, 6, 54, 27, 64, 71, 464, 10, 313, 193, 26, 191, 226, 135, 253, 449, 198, 503, 105, 330, 165, 128, 365, 339, 505, 486]
Qt = control(X_0,X_d,C,N,gamma,omega,n,T)
# mu = optpolicy(Qt)