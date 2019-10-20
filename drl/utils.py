import numpy as np

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
    return sigma * np.random.randn() + mu


def updateQ(a, r, Q, N, alpha=0.1, method="sample_average"):
    if method == "sample_average":
        Q[a] = Q[a] + 1 / float(N[a]) * (r - Q[a])
        return Q

    if method == "constant":
        Q[a] = Q[a] + alpha * (r - Q[a])
        return Q


def is_optimal_action(a, q):
    if (a == np.argmax(q)):  # or Q[a] == Q[np.argmax(Q)]):
        return 1
    else:
        return 0

