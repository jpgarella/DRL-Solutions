import numpy as np
import matplotlib.pyplot as plt

def action_selection(epsilon, k, Q):
    """
    Returns an integer between 0 and k-1.
    """
    random_action = True if np.random.rand() < epsilon else False

    if random_action:
        action = np.random.randint(0, high=k - 1)
    else:
        action = np.argmax(Q)

    return action


def bandit(mu, sigma):
    """
    Return instance of a normal distribution.
    """
    return sigma * np.random.randn() + mu


def updateQ(a, r, Q, N, alpha=0.1, method="sample_average"):
    if method == "sample_average":
        Q[a] = Q[a] + 1. / float(N[a]) * (r - Q[a])
        return Q

    if method == "constant":
        Q[a] = Q[a] + alpha * (r - Q[a])
        return Q


def is_optimal_action(a, q):
    if (a == np.argmax(q)):  # or Q[a] == Q[np.argmax(Q)]):
        return 1
    else:
        return 0


def armed_testbed(k=10, alpha=0.1, method="sample_average", epsilon=0.1,
                  independent_runs=2000, mu0=0, sigma0=1, steps=5000,
                  sigma_bandit=1, mu_random_walk=0, sigma_random_walk=0.01, non_stationary=False,
                  optimistic=0, init_q_equal=False):

    q_ir = []
    Q_ir = []
    R_ir = []
    N_ir = []
    A_ir = []
    Optimal_ir = []

    for ir in range(independent_runs):

        # log status
        if (ir % 100) == 0:
            print("------ " + str(ir) + " / " + str(independent_runs) + " ------")

        # Initialization
        if init_q_equal:
            # Initialize expected reward given a is selected: q (equal to all actions)
            q = np.ones(k) * np.random.normal(mu0, sigma0)
        else:
            # Initialize expected reward given a is selected: q (different for all actions)
            q = sigma0 * np.random.randn(k) + mu0

        if optimistic > 0:
            # Initialize optimistic estimated value of action: Q (equal to all actions)
            Q = np.ones(k) * optimistic
        else:
            # Initialize estimated value of action: Q (equal to all actions)
            Q = np.zeros(k)

        R = []
        N = np.zeros(k)
        A = []
        Optimal = []
        # Loop
        for i in range(steps):
            # epsilon-greedy action selection
            a = action_selection(epsilon, k, Q)
            A.append(a)
            r = bandit(q[a], sigma_bandit)
            R.append(r)
            N[a] += 1
            o = is_optimal_action(a, q)
            Optimal.append(o)
            Q = updateQ(a, r, Q, N, method=method, alpha=alpha)

            if non_stationary:
                # q_a incremental update per step
                q_step_increment = np.random.normal(mu_random_walk, sigma_random_walk, k)
                q = q + q_step_increment

        q_ir.append(q)
        Q_ir.append(Q)
        R_ir.append(np.array(R))
        N_ir.append(N)
        A_ir.append(np.array(A))
        Optimal_ir.append(np.array(Optimal))

    return {'q': np.array(q_ir), 'Q': np.array(Q_ir), 'R': np.array(R_ir), 'N': np.array(N_ir), 'A': np.array(A_ir),
            'O': np.array(Optimal_ir)}

if __name__ == '__main__' :
    # define global variables

    mu0, sigma0 = 0., 1.
    sigma_bandit = 1.

    k = 10
    steps = 2000
    independent_runs = 2000
    method = "sample_average"

    # define local variables for each experiment

    # exp1
    exp1 = {
        'epsilon': 0
    }

    # exp2
    exp2 = {
        'epsilon': 0.01
    }

    # exp3
    exp3 = {
        'epsilon': 0.1
    }


    res1 = armed_testbed(k=k, method=method,
                         epsilon=exp1['epsilon'], independent_runs=independent_runs,
                         mu0=mu0, sigma0=sigma0, steps=steps, sigma_bandit=sigma_bandit,
                         non_stationary=False)

    res2 = armed_testbed(k=k, method=method,
                         epsilon=exp2['epsilon'], independent_runs=independent_runs,
                         mu0=mu0, sigma0=sigma0, steps=steps, sigma_bandit=sigma_bandit,
                         non_stationary=False)

    res3 = armed_testbed(k=k, method=method,
                         epsilon=exp3['epsilon'], independent_runs=independent_runs,
                         mu0=mu0, sigma0=sigma0, steps=steps, sigma_bandit=sigma_bandit,
                         non_stationary=False)

    # plot

    fig = plt.figure(figsize=(16,5))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.plot(range(steps), res1['R'].mean(0),'g')
    axes.plot(range(steps), res2['R'].mean(0), 'r')
    axes.plot(range(steps), res3['R'].mean(0), 'b')
    axes.legend(('$\epsilon$: ' + str(exp1['epsilon']), '$\epsilon$: ' + str(exp2['epsilon']), '$\epsilon$: ' + str(exp3['epsilon'])))
    plt.title('Average reward')

    fig = plt.figure(figsize=(16,5))
    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.plot(range(steps), res1['O'].mean(0)*100, 'g')
    axes.plot(range(steps), res2['O'].mean(0)*100, 'r')
    axes.plot(range(steps), res3['O'].mean(0)*100, 'b')
    axes.legend(('$\epsilon$: ' + str(exp1['epsilon']), '$\epsilon$: ' + str(exp2['epsilon']), '$\epsilon$: ' + str(exp3['epsilon'])))
    plt.title('% Optimal actions')
    plt.show()

