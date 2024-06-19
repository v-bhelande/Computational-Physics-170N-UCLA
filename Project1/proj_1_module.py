import numpy as np

def dynamics_solve(f, D = 1, t_0 = 0.0, s_0 = 1, h = 0.1, N = 100, method = "Euler"):
    
    """ Solves for dynamics of a given dynamical system
    
    - User must specify dimension D of phase space.
    - Includes Euler, RK2, RK4, that user can choose from using the keyword "method"
    
    Args:
        f: A python function f(t, s) that assigns a float to each time and state representing
        the time derivative of the state at that time.
        
    Kwargs:
        D: Phase space dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        s_0: Initial state (float for D=1, ndarray for D>1) set to 1.0 as default
        h: Step size (float) set to 0.1 as default
        N: Number of steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4"
    
    Returns:
        T: Numpy array of times
        S: Numpy array of states at the times given in T
    """
    
    T = np.array([t_0 + n * h for n in range(N + 1)])
    
    if D == 1:
        S = np.zeros(N + 1)
    
    if D > 1:
        S = np.zeros((N + 1, D))
        
    S[0] = s_0
    
    if method == 'Euler':
        for n in range(N):
            S[n + 1] = S[n] + h * f(T[n], S[n])
    
    if method == 'RK2':
        for n in range(N):
            k1 = h * f(T[n], S[n])
            k2 = h * f(T[n], S[n] + (k1/2))
            S[n + 1] = S[n] + k2
    
    if method == 'RK4':
        for n in range(N):
            k1 = h * f(T[n], S[n])
            k2 = h * f(T[n], S[n] + (k1/2))
            k3 = h * f(T[n], S[n] + (k2/2))
            k4 = h * f(T[n], S[n] + k3)
            S[n + 1] = S[n] + (k1/6) + (k2/3) + (k3/3) + (k4/6)
            
    return T, S

def hamiltonian_solve(d_qH, d_pH, d = 1, t_0 = 0.0, q_0 = 0.0, p_0 = 1.0, h = 0.1, N = 100, method = "Euler",):
    
    """ Solves for dynamics of Hamiltonian system
    
    - User must specify dimension d of configuration space.
    - Includes Euler, RK2, RK4, Symplectic Euler (SE) and Stormer Verlet (SV) 
      that user can choose from using the keyword "method"
    
    Args:
        d_qH: Partial derivative of the Hamiltonian with respect to coordinates (float for d=1, ndarray for d>1)
        d_pH: Partial derivative of the Hamiltonian with respect to momenta (float for d=1, ndarray for d>1)
        
    Kwargs:
        d: Spatial dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        q_0: Initial position (float for d=1, ndarray for d>1) set to 0.0 as default
        p_0: Initial momentum (float for d=1, ndarray for d>1) set to 1.0 as default
        h: Step size (float) set to 0.1 as default
        N: Number of steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4", "SE", "SV"
    
    Returns:
        T: Numpy array of times
        Q: Numpy array of positions at the times given in T
        P: Numpy array of momenta at the times given in T
    """

    T = np.array([t_0 + n * h for n in range(N + 1)])
    
    if d == 1:
        P = np.zeros(N + 1)
        Q = np.zeros(N + 1)
        
        Q[0] = q_0
        P[0] = p_0
    
    if d > 1:
        P = np.zeros(N + 1, d)
        Q = np.zeros(N + 1, d)
        
        Q[0] = q_0
        P[0] = p_0
    
    if method == 'Euler':
        for n in range(N):
            Q[n + 1] = Q[n] + h * d_pH(P[n], Q[n])
            P[n + 1] = P[n] + h * (- d_qH(P[n], Q[n]))
    
    if method == 'RK2':
        for n in range(N):
            k1_Q = h * d_pH(P[n], Q[n])
            k1_P = h * (- d_qH(P[n], Q[n]))
            
            k2_Q = h * d_pH(P[n] + (k1_P/2), Q[n] + (k1_Q/2))
            k2_P = h * (- d_qH(P[n] + (k1_P/2), Q[n] + (k1_Q/2)))
            
            Q[n + 1] = Q[n] + k2_Q
            P[n + 1] = P[n] + k2_P
 
    if method == 'RK4':
        for n in range(N): 
            k1_Q = h * d_pH(P[n], Q[n])
            k1_P = h * (- d_qH(P[n], Q[n]))
            
            k2_Q = h * d_pH(P[n] + (k1_P/2), Q[n] + (k1_Q/2))
            k2_P = h * (- d_qH(P[n] + (k1_P/2), Q[n] + (k1_Q/2)))

            k3_Q = h * d_pH(P[n] + (k2_P/2), Q[n] + (k2_Q/2))
            k3_P = h * (- d_qH(P[n] + (k2_P/2), Q[n] + (k2_Q/2)))

            k4_Q = h * d_pH(P[n] + k3_P, Q[n] + k3_Q)
            k4_P = h * (- d_qH(P[n] + k3_P, Q[n] + k3_Q))

            Q[n + 1] = Q[n] + (k1_Q/6) + (k2_Q/3) + (k3_Q/3) + (k4_Q/6)
            P[n + 1] = P[n] + (k1_P/6) + (k2_P/3) + (k3_P/3) + (k4_P/6)

       
    if method == 'SE':
        for n in range(N):
            k_Q = h * d_pH(P[n], Q[n])
            Q[n + 1] = Q[n] + k_Q

            k_P = h * d_qH(P[n], Q[n + 1])
            P[n + 1] = P[n] - k_P

    if method == 'SV':
        for n in range(N):
            temp_P = P[n] - (h/2) * d_qH(P[n], Q[n])
            Q[n + 1] = Q[n] + h * d_pH(temp_P, Q[n])
            P[n + 1] = temp_P - (h/2) * d_qH(P[n], Q[n + 1])
        #return 'TODO!'
        
    return T, Q, P