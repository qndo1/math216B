import numpy as np

def relu(x):
    return np.maximum(0, x)

def compute_activation_matrices(X, W_list, b_list):
    """
    Parameters
    ----------
    X : ndarray of shape (d, N)
        Data matrix whose columns are data points x_i.
    W_list : list of ndarray
        W_list[l] has shape (m_l, m_{l-1}) for layer l = 1..L
    b_list : list of ndarray
        b_list[l] has shape (m_l, 1) for layer l = 1..L

    Returns
    -------
    A_list : list of ndarray
        A_list[l-1] = A_l = Φ_l(X) of shape (m_l, N)
    """
    A_list = []

    Phi = X  # Φ_0(X) = X, shape (d, N)

    for W, b in zip(W_list, b_list):
        # Linear step: Z_l = W_l Φ_{l-1}(X) + b_l
        Z = W @ Phi + b  # broadcasting b over all N columns

        # Nonlinearity: Φ_l(X) = ReLU(Z_l)
        Phi = relu(Z)

        # Store activation matrix A_l
        A_list.append(Phi)

    return A_list
