
# The 10-armed Testbed

# import dependencies
import matplotlib.pyplot as plt
import numpy as np
from drl.utils import action_selection, bandit, updateQ, is_optimal_action

# mean and standard deviation
mu, sigma = 0, 1
mu0, sigma0 = 0, 1
sigma_bandit = 1

# define variables
k = 10
steps = 2000
independent_runs = 2000
alpha = 0.1
epsilons = [0, 0.01, 0.1]

# Initialization
q_ir_e = []
Q_ir_e = []
R_ir_e = []
N_ir_e = []
A_ir_e = []
O_ir_e = []

for e in range(len(epsilons)):

    epsilon = epsilons[e]

    q_ir = []
    Q_ir = []
    R_ir = []
    N_ir = []
    A_ir = []
    O_ir = []
    for ir in range(independent_runs):

        # log status
        if ((ir % 100) == 0):
            print("------ " + str(ir) + " / " + str(independent_runs) + " ------")

        # Initialization
        # q = np.ones(k) * np.random.normal(mu0, sigma0) # Initialize expected reward given a is selected: q (equal to all actions)

        q = sigma0 * np.random.randn(k) + mu0  # Initialize expected reward given a is selected: q (different for all actions)
        Q = np.zeros(k)  # Initialize estimated value of action: Q (equal to all actions)
        R = []
        N = np.zeros(k)
        A = []
        O = []
        # Loop
        for i in range(steps):
            # epsilon-greedy action selection
            a = action_selection(epsilon, k, Q)
            A.append(a)
            r = bandit(q[a], sigma_bandit)
            R.append(r)
            N[a] += 1
            o = is_optimal_action(a, q)
            O.append(o)
            Q = updateQ(a, r, Q, N, method="sample_average")

            # q_a incremental update per step
            # q_t_step_increment = np.random.normal(mu, sigma, k)
            # q_t = q_t + q_t_step_increment

        q_ir.append(q)
        Q_ir.append(Q)
        R_ir.append(np.array(R))
        N_ir.append(N)
        A_ir.append(np.array(A))
        O_ir.append(np.array(O))

    q_ir_e.append(np.array(q_ir))
    Q_ir_e.append(np.array(Q_ir))
    R_ir_e.append(np.array(R_ir))
    N_ir_e.append(np.array(N_ir))
    A_ir_e.append(np.array(A_ir))
    O_ir_e.append(np.array(O_ir))



# plot

fig= plt.figure(figsize=(16,5))
axes= fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(range(steps), R_ir_e[0].mean(0))
axes.plot(range(steps), R_ir_e[1].mean(0))
axes.plot(range(steps), R_ir_e[2].mean(0))
axes.legend(('$\epsilon$: ' + str(epsilons[0]), '$\epsilon$: ' + str(epsilons[1]), '$\epsilon$: ' + str(epsilons[2]) ))
plt.title('Average reward')
plt.show()

fig= plt.figure(figsize=(16,5))
axes= fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(range(steps), O_ir_e[0].mean(0)*100)
axes.plot(range(steps), O_ir_e[1].mean(0)*100)
axes.plot(range(steps), O_ir_e[2].mean(0)*100)
axes.legend(('$\epsilon$: ' + str(epsilons[0]), '$\epsilon$: ' + str(epsilons[1]), '$\epsilon$: ' + str(epsilons[2]) ))
plt.title('% Optimal actions')
plt.show()