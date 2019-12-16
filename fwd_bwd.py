import numpy as np


def rand_nice_probs(shp):
    E = np.random.rand(*shp)
    for i in range(100):
        E = E.round(2)
        E /= E.sum(1, keepdims=True)
    return E

def calc_alpha_k_z_k(E, T, alpha_km1):
    # E shape = [N_possible_current_states]
    # T shape = [N_possible_previous_states, N_possible_current_states]
    # ak1 shp = [N_possible_previous_states]
    v = (T * alpha_km1[:, None]).sum(0)
    return (E * v)

def forward(E, T, init_state, observations):
    fwds = [init_state * E[:, observations[0]]]
    for i, obs in enumerate(observations[1:]):
        fwds.append(calc_alpha_k_z_k(E[:, obs], T, fwds[-1]))
    return np.array(fwds)

def calc_beta_k_z_k(E, T, beta_kp1):
    # E shape = [N_possible_future_states]
    # T shape = [N_possible_current_states, N_possible_future_states]
    # bk1 shp = [N_possible_future_states]
    v = (T * beta_kp1[None, :] * E[None, :]).sum(1)
    return v

def backward(E, T, init_state, observations):
    bwds = [np.ones(T.shape[0])]
    for i, obs in enumerate(reversed(observations[1:])):
        bwds.insert(0, calc_beta_k_z_k(E[:, obs], T, bwds[0]))
    return np.array(bwds)


def forward_backward(E, T, init_state, observations):
    assert np.all(E.sum(1) == 1)
    assert np.all(T.sum(1) == 1)
    fwd = forward(E, T, init_state, observations)
    bak = backward(E, T, init_state, observations)
    assert np.abs((bak[0] * init_state * E[:, observations[0]])[:2].sum() - fwd[-1][:2].sum()) < 0.0001
    norm_const = fwd[-1][:2].sum()
    return (fwd * bak) / norm_const


if __name__ == "__main__":
    T = rand_nice_probs((2, 2))
    E = rand_nice_probs((2, 3))
    E[1, 1] = 5
    E[0, 1] = 0.1
    E = np.array([[0.30, 0.1, 0.60],
                  [0.08, 0.9, 0.02]])
    E /= E.sum(1, keepdims=True)
    T = np.array([[0.9,  0.1 ],
                  [0.05, 0.95]])
    print(T)
    print(E)

    init_state = np.array([0.99, 0.01])
    observations = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    forward_backward(E, T, init_state, observations)
