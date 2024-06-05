import numpy as np

def mi_estimator(mu, X, k, gamma):
    """
    Mutual Information Estimator

    Parameters:
    mu : function
        A function that returns the probability of a given tuple.
    X : np.ndarray
        Sampled tuples, shape (k, n).
    k : int
        Sample size.
    gamma : float
        Stabilization parameter.

    Returns:
    float
        MI estimate.
    """
    # Step 2: Independently sample tuples X1, ..., Xk ~ mu
    # (Already given as input X)

    # Step 3: Construct a set of indices of unique elements S
    S = []
    seen = set()
    for i in range(k):
        tuple_i = tuple(X[i])
        if tuple_i not in seen:
            S.append(i)
            seen.add(tuple_i)

    # Step 4: Construct empirical distributions
    Z = sum(mu(tuple(X[i])) for i in S)
    
    def mu_hat(Xi):
        return mu(Xi) / Z
    
    def mu_hat_tensor(x_prime, j):
        numerator = 1
        for i in range(len(x_prime)):
            if i != j:
                numerator *= sum(mu(tuple(np.concatenate((x_prime[:i], [xi], x_prime[i+1:])))) for xi in X[:, i]) / Z
        return numerator
    
    Z_tensor = sum(mu(tuple(np.concatenate((X[i][:j], [X[j][j]], X[i][j+1:]))) for i in S for j in range(len(X[0]))))

    def mu_hat_tensor_for_all_x_prime(Xi):
        return mu_hat_tensor(Xi, Xi.index(Xi)) / Z_tensor

    # Step 5: Compute estimate
    I_hat_k_gamma = 0
    for i in S:
        Xi = tuple(X[i])
        I_hat_k_gamma += mu_hat(Xi) * np.log((mu_hat(Xi) + gamma / Z) / (mu_hat_tensor_for_all_x_prime(Xi) + gamma / Z_tensor))
    
    return I_hat_k_gamma

# Example usage
def example_mu(x):
    # Define a dummy probability function
    return np.exp(-np.sum(np.square(x)))

# Generate sample data
k = 10
n = 3
X = np.random.randn(k, n)
gamma = 1 / k

# Compute MI estimate
mi_estimate = mi_estimator(example_mu, X, k, gamma)
print("MI Estimate:", mi_estimate)
