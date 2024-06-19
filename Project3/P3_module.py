import numpy as np

# 1. Function Definition
def weighted_die(num_steps):

    """
    Args:
        num_steps: Number of MCMC updates to perform

    6 sided dice
    if p_k denotes the probability that side k will land face up, then:
    1) p_1 = p_2
    2) p_3 = p_4 = p_5 = p_6
    3) p_1 = 3p_3

    Returns:
        outcomes: Numpy array containing states of die at each step
        my_dollars: Your winnings from the game
        friend_dollars: Friend's winnings from the game
    """

    # List to store outcomes from different die rolls
    outcomes = np.zeros(num_steps, dtype=np.int)

    # Possible Outcomes
    states = np.array([1, 2, 3, 4, 5, 6], dtype=np.int)

    # Probabilities: [p_1, p_2, p_3, p_4, p_5, p6]
    prior_probs = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])

    # 1) Pick initial state s_1
    init_state = np.random.choice(states, p=prior_probs)
    init_state = init_state.astype(int)
    outcomes[0] = init_state

    # Assign initial state as current state
    curr_state = outcomes[0]
    curr_state = curr_state.astype(int)

    # 4) Repeat steps 2 and 3 for num_steps
    for step_num in range(1, num_steps):

        # 2) Propose new state s' according to distribution q(s'|s)
        new_state = np.random.choice(states, p=prior_probs)
        new_state = new_state.astype(int)

        # 3) Calculate A(s'|s)
        # In this case, we know q(s'|s) = q(s|s')
        prior_prob_curr_state = prior_probs[curr_state-1]
        prior_prob_new_state = prior_probs[new_state-1]
        A = min(1, prior_prob_new_state/prior_prob_curr_state)

        # Accept new state if its probability is greater than the current state
        if A == 1:
            outcomes[step_num] = new_state
        # Otherwise run the probability that the new state is the next state
        else:
            outcomes[step_num] = np.random.choice([new_state, curr_state], p=[A, 1.-A])
        # Assign new state as current state
        curr_state = outcomes[step_num]
        curr_state = curr_state.astype(int)

    # Calculate my winnings and friend's winnings
    my_dollars = 0
    friend_dollars = 0

    for state in outcomes:
        if state == 1 or state == 2: my_dollars += 1
        else: friend_dollars += 1

    return outcomes, my_dollars, friend_dollars

# 2. Function Definitions
def find_energy(lattice, L, s_ij, i, j, H=0):
    if i == 0: left_index = L-1
    else: left_index = i-1

    if i == L-1: right_index = 0
    else: right_index = i+1

    if j == 0: top_index = L-1
    else: top_index = j-1

    if j == L-1: bottom_index = 0
    else: bottom_index = j+1

    s_top = lattice[i][top_index]
    s_bottom = lattice[i][bottom_index]
    s_left = lattice[left_index][j]
    s_right = lattice[right_index][j]

    return -s_ij*(s_top+s_bottom+s_left+s_right) - H*(s_top+s_bottom+s_left+s_right)

def energy_change(lattice, L, s_ij, i, j, H=0):
    if i == 0: left_index = L-1
    else: left_index = i-1

    if i == L-1: right_index = 0
    else: right_index = i+1

    if j == 0: top_index = L-1
    else: top_index = j-1

    if j == L-1: bottom_index = 0
    else: bottom_index = j+1

    s_top = lattice[i][top_index]
    s_bottom = lattice[i][bottom_index]
    s_left = lattice[left_index][j]
    s_right = lattice[right_index][j]

    return 2*s_ij*(s_top+s_bottom+s_left+s_right+H)

def find_S(lattice):
    return np.sum(lattice)

def find_new_state(lattice, L, s_ij, i, j, temp):
    curr_state = lattice[i][j]
    other_state = -1.*curr_state
    A = min(1, np.exp(-energy_change(lattice, L, s_ij, i, j)/temp))
    new_state = np.random.choice([other_state, curr_state], p=[A, 1.-A])
    return new_state

# 2.1 Function Definition
def two_dim_ising(L, temp, num_steps):

    """
    Args:
        L: Specified side length (Number of spins)
        temp: Desired temperature of lattice
        num_steps: Number of MCMC updates to perform

    Returns:
        lattice: L x L Numpy array with updated lattice values
        U: Numpy array of mean internal energy at each step
        M: Numpy array of magnetization at each step
        t: Numpy array of updating step per site defined as: Total updates/Number of sites
    """

    # Define critcal temperature
    T_c = 2.2692 #K

    N = L**2

    # Generate L x L lattice with s = -1, 1
    spins = np.array([-1, 1])
    lattice = np.zeros((L, L))
    for i, row in enumerate(lattice):
        for j in range(len(row)):
            lattice[i][j] = np.random.choice(spins)

    # Initialize <E>, <S>
    exp_E = []
    exp_S = []

    # Repeat for N steps
    for step in range(num_steps):

        # Pick a random site i on the 2D lattice and compute the energy change due to the change of sign in s_i
        row = np.random.choice(range(L))
        col = np.random.choice(range(L))
        s = lattice[row][col]

        # In the first step, we just choose an initial site and find the energy
        # The expectation value of energy is equal to that energy
        if step == 0:
            energy = find_energy(lattice, L, s, row, col)
            exp_E.append(energy)
            magnetization = find_S(lattice)
            exp_S.append(magnetization)
            continue

        delta_E = energy_change(lattice, L, s, row, col)
        OLD_ENERGY = find_energy(lattice, L, s, row, col)

        # If delta_E <= 0 accept move and flip spin sign
        if delta_E <= 0:
            lattice[row][col] *= -1.
            delta_S = 2.*lattice[row][col]
        # Else accept move and flip spin sign with probability A
        else:
            new_state = find_new_state(lattice, L, s, row, col, temp)
            if lattice[row][col] != new_state and new_state == -1.:
                lattice[row][col] = new_state
                delta_S = -2.
            elif lattice[row][col] != new_state and new_state == 1.:
                lattice[row][col] = new_state
                delta_S = 2.
            else:
                delta_E = 0.  # Proposed energy change will not occur!
                delta_S = 0.

        NEW_ENERGY = find_energy(lattice, L, lattice[row][col], row, col)

        # Update <E>, <S> and append them to lists
        energy += delta_E
        assert(np.isclose(NEW_ENERGY-OLD_ENERGY, delta_E))
        exp_E_new = exp_E[step-1] + (energy-exp_E[step-1])/(step)
        exp_E.append(exp_E_new)
        
        magnetization += delta_S
        exp_S_new = exp_S[step-1] + (magnetization-exp_S[step-1])/(step)
        exp_S.append(exp_S_new)

    # Calculate U, M, t
    U = np.array(exp_E)/N
    M = np.array(exp_S)/N
    t = np.arange(1, num_steps+1)/N
        
    return lattice, U, M, t

# 2.2 Function Defintion
def two_dim_ising2(L, temp, num_steps, lattice=None):

    """
    Args:
        L: Specified side length (Number of spins)
        temp: Desired temperature of lattice
        num_steps: Number of MCMC updates to perform
        lattice (optional): Initial configuration of lattice

    Returns:
        lattice: L x L Numpy array with updated lattice values
        M: Numpy array of magnetization at each step
    """

    # Define critcal temperature
    T_c = 2.2692 #K

    N = L**2

    # Generate L x L lattice with s = -1, 1
    if lattice is None:
        spins = np.array([-1, 1])
        lattice = np.zeros((L, L))
        for i, row in enumerate(lattice):
            for j in range(len(row)):
                lattice[i][j] = np.random.choice(spins)

    # Repeat for N steps
    for step in range(num_steps):

        # Pick a random site i on the 2D lattice and compute the energy change due to the change of sign in s_i
        row = np.random.choice(range(L))
        col = np.random.choice(range(L))
        s = lattice[row][col]

        # In the first step, we just choose an initial site and find the energy
        # The expectation value of energy is equal to that energy
        if step == 0:
            magnetization = find_S(lattice)
            exp_S = magnetization
            continue

        delta_E = energy_change(lattice, L, s, row, col)

        # If delta_E <= 0 accept move and flip spin sign
        if delta_E <= 0:
            lattice[row][col] *= -1.
            delta_S = 2.*lattice[row][col]
        # Else accept move and flip spin sign with probability A
        else:
            new_state = find_new_state(lattice, L, s, row, col, temp)
            if lattice[row][col] != new_state and new_state == -1.:
                lattice[row][col] = new_state
                delta_S = -2.
            elif lattice[row][col] != new_state and new_state == 1.:
                lattice[row][col] = new_state
                delta_S = 2.
            else:
                delta_E = 0.  # Proposed energy change will not occur!
                delta_S = 0.

        # Update <S> and append to list
        magnetization += delta_S
        exp_S_new = exp_S + (magnetization-exp_S)/(step)
        exp_S = exp_S_new

    # Calculate M
    M = exp_S/N
        
    return lattice, M

# 2.3 Function Definition
def two_dim_ising3(L, temp, num_steps):

    """
    Args:
        L: Specified side length (Number of spins)
        temp: Desired temperature of lattice
        num_steps: Number of MCMC updates to perform

    Returns:
        lattice: L x L Numpy array with updated lattice values
    """

    # Define critcal temperature
    T_c = 2.2692 #K

    N = L**2

    # Generate L x L lattice with s = -1, 1
    spins = np.array([-1, 1])
    lattice = np.zeros((L, L))
    for i, row in enumerate(lattice):
        for j in range(len(row)):
            lattice[i][j] = np.random.choice(spins)

    # Repeat for N steps
    for step in range(num_steps):

        # Pick a random site i on the 2D lattice and compute the energy change due to the change of sign in s_i
        row = np.random.choice(range(L))
        col = np.random.choice(range(L))
        s = lattice[row][col]

        delta_E = energy_change(lattice, L, s, row, col)

        # If delta_E <= 0 accept move and flip spin sign
        if delta_E <= 0:
            lattice[row][col] *= -1.
        # Else accept move and flip spin sign with probability A
        else:
            new_state = find_new_state(lattice, L, s, row, col, temp)
            if lattice[row][col] != new_state and new_state == -1.:
                lattice[row][col] = new_state
            elif lattice[row][col] != new_state and new_state == 1.:
                lattice[row][col] = new_state
            else: continue
        
    return lattice