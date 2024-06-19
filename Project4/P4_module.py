import numpy as np

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

def two_dim_ising(L, temp, num_steps):

    """
    Args:
        L: Specified side length (Number of spins)
        temp: Desired temperature of lattice
        num_steps: Number of MCMC updates to perform

    Returns:
        lattice: L x L Numpy array with updated lattice values
        temp: Desired temperature of lattice
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
        
    return lattice, temp