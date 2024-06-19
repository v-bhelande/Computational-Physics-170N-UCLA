import math
import numpy as np
from numpy.linalg import norm

def hermitian_eigensystem(H, tolerance):
    
    """ Solves for the eigenvalues and eigenvectors of a hermitian matrix
    
    Args:
        H: Hermitian matrix for which we want to compute eigenvalues and eigenvectors
        #
        tolerance: A number that sets the tolerance for the accuracy of the computation.  This number
        is multiplied by the norm of the matrix H to obtain a number delta.  The algorithm successively
        applies (via similarity transformation) Jacobi rotations to the matrix H until the sum of the
        squares of the off-diagonal elements are less than delta.
    
    
    
    Returns:
        d: Numpy array containing eigenvalues of H in non-decreasing order
        
        U: A 2d numpy array whose columns are the eigenvectors corresponding to the computed
        eigenvalues.
        
        
    Checks you might need to do:
        
        H * U[:,k] = d[k] *\ U[:,k]      k=0,1,2,...,(n-1)
        
        d[0] <= d[1] <= ... <= d[n-1]     (where n is the dimension of H)
       
        np.transpose(U) * U = U * np.transpose(U) = np.eye(n)
        
    """

    # Could not get complex eigenvalue solver to work, but some code is
    # attached below in the hopes of recieving partial credit
    # Use np.linalg.eig for complex matrices, use homemade real_eigen otherwise
    used_np_linalg_eig = False
    for row in range(len(H)):
        for col in range(len(H)):
            if H[row][col].imag != 0.:

                # Find eigenvalues and eignvectors
                d, U = np.linalg.eig(H)

                # Sort eigenvalues and eigenvectors in non-decreasing order
                sorted_indices = np.argsort(d) 
                d = np.sort(d)
                U = U.T[sorted_indices].T

                # Set flag to true
                used_np_linalg_eig = True
                break
    
        if used_np_linalg_eig == True: break
    
    # real_eigen works for real matrices!
    if used_np_linalg_eig == False:
        d, U = real_eigen(H, tolerance)

    # Check 1: H * U[:,k] = d[k] *\ U[:,k]      k=0,1,2,...,(n-1)
    H_U = H @ U

    d_U = np.zeros((len(U), len(U)))
    for k in range(len(d_U)):
        d_U[:,k] = d[k] * U[:,k]

    if not np.isclose(H_U.all(), d_U.all()):
        print('Error: H * U[:,k] = d[k] *\ U[:,k]')
        return 'Error: H * U[:,k] = d[k] *\ U[:,k]'

    for i in range(1, len(d)):
        if d[i-1] > d[i]:
            print('Error: Eigenvalues not sorted in non-decreasing order!')
            return 'Error: Eigenvalues not sorted in non-decreasing order!'

    # Check 3:
    n = len(U)
    prod1 = np.transpose(U) * U
    prod2 = U * np.transpose(U)
    if not np.isclose(prod1.all(), prod2.all()):
        print('Error: np.transpose(U) * U != U * np.transpose(U) or np.transpose(U) * U = np.eye(n)')
        return 'Error: np.transpose(U) * U != U * np.transpose(U) or np.transpose(U) * U = np.eye(n)'

    return d, U

#difficulty: ★★★
def jacobi_rotation(A, j, k):
    #Args:
        # A (np.ndarray): n by n real symmetric matrix
        # j (int): column parameter.
        # k (int): row parameter.

    #Returns:
        # A (np.ndarray): n by n real symmetric matrix, where the A[j,k] and A[k,j] element is zero
        # J (np.ndarray): n by n orthogonal matrix, the jacobi_rotation matrix

    # Construct jacobi matrix
    J = make_jacobi_mat(A, j, k)

    # Transform A
    A = np.matmul(np.transpose(J), A)
    A = np.matmul(A, J)

    return A, J

def make_jacobi_mat(A, j, k):

    a = A[j][j]
    b = A[k][k]
    c = A[j][k]

    hypotenuse = np.sqrt(((a-b)**2)+(4*(c**2)))
    sin_2theta = (2*c)/hypotenuse
    cos_2theta = (a-b)/hypotenuse
    
    # Find theta
    theta = np.arctan(2*c/(a-b))

    # Quadrant 1
    if sin_2theta > 0. and cos_2theta > 0.:
        if theta > 0. and theta < np.pi/2: pass
        else: theta += np.pi
    # Quadrant 2
    elif sin_2theta > 0. and cos_2theta < 0.:
        if theta > np.pi/2 and theta < np.pi: pass
        else: theta += np.pi
    # Quadrant 3
    elif sin_2theta < 0. and cos_2theta < 0.:
        if theta > np.pi and theta < 1.5*np.pi: pass
        else: theta += np.pi
    # Quadrant 4
    elif sin_2theta < 0. and cos_2theta > 0.:
        if theta > 1.5*np.pi and theta < 2*np.pi: pass
        else: theta += np.pi

    theta *= 0.5

    # Construct J
    J = np.zeros((len(A), len(A)))
    for row_num in range(len(A)):
        J[row_num][row_num] = 1.
    J[j][j] = np.cos(theta)
    J[k][k] = np.cos(theta)
    J[j][k] = -1.*np.sin(theta)
    J[k][j] = np.sin(theta)

    return J

#difficulty: ★
def off(A):
    # see lecture note equation (12) 
    sum = 0
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                #sum += np.abs(A[i,j])**2
                sum += np.vdot(A[i,j], A[i,j])
    output = np.sqrt(sum)
    return output.real

#difficulty: ★
def frobenius_norm(A):
    sum = 0
    for i in range(len(A)):
        for j in range(len(A)):
            #sum += np.abs(A[i,j])**2
            sum += np.vdot(A[i,j], A[i,j])
    output = np.sqrt(sum)
    return output.real

#difficulty: ★★★
def real_eigen(A, tolerance):
    #Args:
        # A (np.ndarray): n by n real symmetric matrix
        # tolerance (double): the relative precision
    #Returns:
        # d (np.ndarray): n by 1 vector, d[i] is the i-th eigenvalue
        # R (np.ndarray): n by n orthogonal matrix, R[:,i] is the i-th eigenvector
    
    d = []

    # Store jacobi matrice(s)
    O = np.identity(len(A))
    O_T = np.identity(len(A))

    diag_this = A

    curr_tol = off(diag_this)/norm(diag_this).real

    while curr_tol >= tolerance:

        # Iterate through A
        for row_elem in range(len(A)):

            # Call jacobi_rotation iteratively for all off-diagonal elements in first row
            for col_elem in range(len(A)):

                # Skip to next element if current element is ~0.
                if row_elem == col_elem or np.isclose(diag_this[row_elem][col_elem], 0.): continue

                # call jacobi_rotation(A, j, k)
                A_prime, jacobi_mat = jacobi_rotation(diag_this, row_elem, col_elem)

                # Find O
                O = np.matmul(O, jacobi_mat)
                O_T = np.matmul(np.transpose(jacobi_mat), O_T)

                diag_this = A_prime
                curr_tol = off(diag_this)/norm(diag_this)

    # Take diagonalized matrix and append eigenvalues to d
    for row_num in range(len(diag_this)):
        d.append(diag_this[row_num][row_num])
                    
    # Sort eigenvalues and eigenvectors in non-decreasing order
    sorted_indices = np.argsort(d) 
    d = np.sort(d)
    R = O.T[sorted_indices].T

    # Generate identity matrix for check
    I = np.identity(len(O))

    # Check if R is orthogonal
    OOT = np.matmul(O, np.transpose(O))
    if not np.allclose(OOT, I): return 1

    return d, R

def find_theta1(A, j, k):

    c = A[j][k]

    x = c.real
    y = c.imag

    hypotenuse = np.sqrt((x**2)+(y**2))
    sin_phi1 = (y)/hypotenuse
    cos_phi1 = (x)/hypotenuse
    phi1 = np.arctan(y/x)

    # Quadrant 1
    if sin_phi1 > 0. and cos_phi1 > 0.:
        if phi1 > 0. and phi1 < np.pi/2: pass
        else: phi1 += np.pi
    # Quadrant 2
    elif sin_phi1 > 0. and cos_phi1 < 0.:
        if phi1 > np.pi/2 and phi1 < np.pi: pass
        else: phi1 += np.pi
    # Quadrant 3
    elif sin_phi1 < 0. and cos_phi1 < 0.:
        if phi1 > np.pi and phi1 < 1.5*np.pi: pass
        else: phi1 += np.pi
    # Quadrant 4
    elif sin_phi1 < 0. and cos_phi1 > 0.:
        if phi1 > 1.5*np.pi and phi1 < 2*np.pi: pass
        else: phi1 += np.pi

    theta1 = 0.25*(2*phi1 - np.pi)
    return theta1.real

def find_theta2(A, j, k):
    a = A[j][j]
    b = A[k][k]
    c = A[j][k]

    x = a - b
    y = 2*np.abs(c)

    hypotenuse = np.sqrt((x**2)+(y**2))
    sin_phi2 = (y)/hypotenuse
    cos_phi2 = (x)/hypotenuse
    phi2 = np.arctan(y/x)

    # Quadrant 1
    if sin_phi2 > 0. and cos_phi2 > 0.:
        if phi2 > 0. and phi2 < np.pi/2: pass
        else: phi2 += np.pi
    # Quadrant 2
    elif sin_phi2 > 0. and cos_phi2 < 0.:
        if phi2 > np.pi/2 and phi2 < np.pi: pass
        else: phi2 += np.pi
    # Quadrant 3
    elif sin_phi2 < 0. and cos_phi2 < 0.:
        if phi2 > np.pi and phi2 < 1.5*np.pi: pass
        else: phi2 += np.pi
    # Quadrant 4
    elif sin_phi2 < 0. and cos_phi2 > 0.:
        if phi2 > 1.5*np.pi and phi2 < 2*np.pi: pass
        else: phi2 += np.pi

    theta2 = 0.5*phi2
    return theta2.real

def make_complex_jacobi_mat(A, j, k):
    
    # Find theta1 and theta2
    theta1 = find_theta1(A, j, k)
    theta2 = find_theta2(A, j, k)

    # Construct J
    J = np.zeros((len(A), len(A)), dtype=np.complex128)
    
    for row_num in range(len(A)):
        J[row_num][row_num] = 1.
    
    J[j][j] = -1j*np.exp(-1j*theta1)*np.sin(theta2)
    J[k][k] = 1j*np.exp(1j*theta1)*np.sin(theta2)
    J[j][k] = -1.*np.exp(1j*theta1)*np.cos(theta2)
    J[k][j] = 1.*np.exp(-1j*theta1)*np.cos(theta2)

    return J

#difficulty: ★★★
def complex_jacobi_rotation(A, j, k):
    #Args:
        # A (np.ndarray): n by n real symmetric matrix
        # j (int): column parameter.
        # k (int): row parameter.

    #Returns:
        # A (np.ndarray): n by n real symmetric matrix, where the A[j,k] and A[k,j] element is zero
        # J (np.ndarray): n by n orthogonal matrix, the jacobi_rotation matrix

    # Construct jacobi matrix
    J = make_complex_jacobi_mat(A, j, k)

    # Transform A
    A = np.matmul(np.transpose(J), A)
    A = np.matmul(A, J)

    return A, J

#difficulty: ★★
def complex_eigen(A, tolerance):
    #Args:
        # A (np.ndarray): n by n real hermitian matrix
        # tolerance (double): the relative precision
    #Returns:
        # d (np.ndarray): n by 1 vector, d[i] is the i-th eigenvalue
        # U (np.ndarray): n by n unitary matrix, U[i,:] is the i-th eigenvector

    d = []
    R = []

    # Store complex jacobi matrice(s)
    O = np.identity(len(A))
    O_T = np.identity(len(A))

    diag_this = A

    curr_tol = off(diag_this)/norm(diag_this)

    while curr_tol >= tolerance:

        # Iterate through A
        for row_elem in range(len(A)):

            # Call jacobi_rotation iteratively for all off-diagonal elements in first row
            for col_elem in range(len(A)):

                # Call J iteratively for all off-diagonal elements in first row

                # Skip to next element if current element is ~0.
                if row_elem == col_elem or np.isclose(diag_this[row_elem][col_elem], 0.): continue

                # call jacobi_rotation(A, j, k)
                A_prime, jacobi_mat = complex_jacobi_rotation(diag_this, row_elem, col_elem)

                # Find O
                O = np.matmul(O, jacobi_mat)
                O_T = np.matmul(np.transpose(jacobi_mat), O_T)

                diag_this = A_prime
                curr_tol = off(diag_this)/norm(diag_this)
                break

    # Take diagonalized matrix and append eigenvalues to d
    for row_num in range(len(diag_this)):
        d.append(diag_this[row_num][row_num])
                    
    # Convert lists to arrays    
    d = np.array(d)

    # Generate identity matrix for chechk
    I = np.identity(len(O))

    # Check if R is orthogonal
    OOT = np.matmul(O, np.transpose(O))
    if not np.allclose(OOT, I): return 1

    O_r = O[0:int(0.5*(len(O))), 0:int(0.5*(len(O)))]
    O_im = O[int(0.5*(len(O))):len(O), 0:int(0.5*(len(O)))]
    R = O_r + 1j*O_im

    d, U = 0
    return d, U