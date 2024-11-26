import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.stats import multivariate_normal
from scipy.sparse.linalg import spsolve

def make_tridiag(a, b, c):
    """
    Create a sparse tridiagonal matrix given a, b, c.
    
    Parameters:
    a (array-like): Subdiagonal elements.
    b (array-like): Main diagonal elements.
    c (array-like): Superdiagonal elements.
    
    Returns:
    scipy.sparse.coo_matrix: Tridiagonal matrix in sparse format.
    """
    N = len(a)
    if len(b) != N or len(c) != N:
        raise ValueError("Make sure the length of a, b, c are the same...")
    
    # Defining indices for sparse matrix
    x_indx = np.concatenate((np.arange(1, N), np.arange(N), np.arange(N - 1)))
    y_indx = np.concatenate((np.arange(N - 1), np.arange(N), np.arange(1, N)))
    eles = np.concatenate((a[1:], b, c[:-1]))
    
    # Creating sparse matrix in coordinate format
    Mat = coo_matrix((eles, (x_indx, y_indx)), shape=(N, N))
    
    return Mat

def tridiag_solver(a, b, c, r):
    """
    Solve Ax = r with a, b, c being the lower, middle, and upper diagonal entries.
    
    Parameters:
    a (array-like): Subdiagonal elements.
    b (array-like): Main diagonal elements.
    c (array-like): Superdiagonal elements.
    r (array-like): Right-hand side vector.
    
    Returns:
    numpy.ndarray: Solution vector x.
    """
    if b[0] == 0:
        raise ValueError("Reorder the equations for the tridiagonal solver...")
    
    N = len(a)
    u = np.zeros(N)
    gamma = np.zeros(N)
    
    # Forward substitution
    beta = b[0]
    u[0] = r[0] / beta
    for j in range(1, N):
        gamma[j] = c[j - 1] / beta
        beta = b[j] - a[j] * gamma[j]
        if beta == 0:
            raise ValueError("The tridiagonal solver failed...")
        u[j] = (r[j] - a[j] * u[j - 1]) / beta
    
    # Back substitution
    for j in range(N - 1, 0, -1):
        u[j - 1] -= gamma[j] * u[j]
    
    return u

def make_matrix(M1, N1, M2, N2, x1_dt, x2_dt, D):
    """
    Creates matrices to solve advection-diffusion problem.
    
    Parameters:
    M1, M2 (float): Specifies the spatial domain x1 = (-M1, M1), x2 = (-M2, M2).
    N1, N2 (int): Grid sizes (size of matrices (N1*N2) x (N1*N2)).
    x1_dt, x2_dt (array-like): Coefficients for advection terms.
    D (array-like): Diffusion coefficients.
    
    Returns:
    tuple: D1, D2, D11, D22, x1, x2 matrices and vectors used to solve the equation.
    """
    dx1 = 2 * M1 / (N1 - 1)
    if dx1 < 0 or dx1 > 0.5:
        raise ValueError("Make M1 smaller and/or N1 larger!!")
    x1 = np.linspace(-M1, M1, N1)
    
    dx2 = 2 * M2 / (N2 - 1)
    if dx2 < 0 or dx2 > 0.5:
        raise ValueError("Make M2 smaller and/or N2 larger!!")
    x2 = np.linspace(-M2, M2, N2)
    
    # Creating D1 matrix
    a, b, c = [], np.zeros(N1 * N2), []
    for i in range(N2):
        a_coeff, c_coeff = 0, 0
        for k in range(len(x1_dt)):
            a_coeff -= (-x1_dt[k, 0]) * (x2[i] ** x1_dt[k, 2]) * (x1[:-1] ** x1_dt[k, 1]) / (2 * dx1)
            c_coeff += (-x1_dt[k, 0]) * (x2[i] ** x1_dt[k, 2]) * (x1[1:] ** x1_dt[k, 1]) / (2 * dx1)
        a.extend([0] + a_coeff.tolist())
        c.extend(c_coeff.tolist() + [0])
    
    D1 = make_tridiag(np.array(a), b, np.array(c))
    
    # Creating D2 matrix
    a, b, c = [], np.zeros(N1 * N2), []
    for i in range(N1):
        a_coeff, c_coeff = 0, 0
        for k in range(len(x2_dt)):
            a_coeff -= (-x2_dt[k, 0]) * (x2[:-1] ** x2_dt[k, 2]) * (x1[i] ** x2_dt[k, 1]) / (2 * dx2)
            c_coeff += (-x2_dt[k, 0]) * (x2[1:] ** x2_dt[k, 2]) * (x1[i] ** x2_dt[k, 1]) / (2 * dx2)
        a.extend([0] + a_coeff.tolist())
        c.extend(c_coeff.tolist() + [0])
    
    D2 = make_tridiag(np.array(a), b, np.array(c))
    
    # Creating D11 matrix
    a, b, c = [], [], []
    for i in range(N2):
        a.extend([0] + (D[0] * (np.ones(N1 - 1) / (dx1 * dx1))).tolist())
        c.extend((D[0] * (np.ones(N1 - 1) / (dx1 * dx1))).tolist() + [0])
        b.extend((-2 * D[0] * (np.ones(N1) / (dx1 * dx1))).tolist())
    
    D11 = make_tridiag(np.array(a), np.array(b), np.array(c))
    
    # Creating D22 matrix
    a, b, c = [], [], []
    for i in range(N1):
        a.extend([0] + (D[1] * (np.ones(N2 - 1) / (dx2 * dx2))).tolist())
        c.extend((D[1] * (np.ones(N2 - 1) / (dx2 * dx2))).tolist() + [0])
        b.extend((-2 * D[1] * (np.ones(N2) / (dx2 * dx2))).tolist())
    
    D22 = make_tridiag(np.array(a), np.array(b), np.array(c))
    
    return D1, D2, D11, D22, x1, x2

def solve_advDif(D1, D2, D11, D22, x1, x2, dt, T_end, mu, sigma):
    """
    Solves the advection-diffusion equation using given matrices.
    
    Parameters:
    D1, D2, D11, D22 (sparse matrices): Matrices to solve advection-diffusion.
    x1, x2 (array-like): Spatial grid points.
    dt (float): Time step.
    T_end (float): End time.
    mu (array-like): Mean for initial Gaussian distribution.
    sigma (array-like): Covariance matrix for initial Gaussian distribution.
    
    Returns:
    t (numpy.ndarray): Time steps.
    p (numpy.ndarray): Solution of the advection-diffusion equation.
    """
    t = np.arange(0, T_end + dt, dt)
    N1 = len(x1)
    N2 = len(x2)
    p = np.zeros((N1, N2, len(t)))
    
    # Initial condition using multivariate normal distribution
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))
    p[:, :, 0] = multivariate_normal(mean=mu, cov=sigma).pdf(pos).T
    
    # Reshape p for time iteration
    p = p.reshape(N1 * N2, len(t))
    
    # Create matrix A and vector b (Diffusion + Drift)
    rhs_first = eye(N1 * N2) + 0.5 * dt * (D1 + D11)
    rhs_second = eye(N1 * N2) + 0.5 * dt * (D2 + D22)
    
    a_first = np.full(N1 * N2, -0.5 * dt * (D2.data[0] + D22.data[0]))
    b_first = np.ones(N1 * N2) - 0.5 * dt * (D2.data[1] + D22.data[1])
    c_first = np.full(N1 * N2, -0.5 * dt * (D2.data[2] + D22.data[2]))
    
    a_second = np.full(N1 * N2, -0.5 * dt * (D1.data[0] + D11.data[0]))
    b_second = np.ones(N1 * N2) - 0.5 * dt * (D1.data[1] + D11.data[1])
    c_second = np.full(N1 * N2, -0.5 * dt * (D1.data[2] + D11.data[2]))
    
    for j in range(1, len(t)):
        if j % 100 == 0:
            print(f'Number of Iteration {j} out of {len(t)} finished...')
        
        # Solve for p[:, j]
        reshaped_r = rhs_first.dot(p[:, j - 1]).reshape(N1, N2).T.reshape(N1 * N2, 1)
        p_half = tridiag_solver(a_first, b_first, c_first, reshaped_r.flatten())
        
        reshaped_r = rhs_second.dot(p_half).reshape(N2, N1).T.reshape(N1 * N2, 1)
        p[:, j] = tridiag_solver(a_second, b_second, c_second, reshaped_r.flatten())
    
    # Reshape p to original form
    p = p.reshape(N1, N2, len(t))
    
    return t, p

# # Example usage
# a = np.array([0, 1, 1, 1])  # Example values for subdiagonal
# b = np.array([2, 2, 2, 2])  # Example values for main diagonal
# c = np.array([3, 3, 3, 0])  # Example values for superdiagonal
# r = np.array([5, 5, 5, 5])  # Example values for RHS vector

# Mat = make_tridiag(a, b, c)
# print(Mat.toarray())

# u = tridiag_solver(a, b, c, r)
# print(u)

# # Example usage for make_matrix
# M1, N1, M2, N2 = 1.0, 4, 1.0, 4
# x1_dt = np.array([[1, 2, 3], [4, 5, 6]])
# x2_dt = np.array([[1, 2, 3], [4, 5, 6]])
# D = np.array([0.1, 0.2])

# D1, D2, D11, D22, x1, x2 = make_matrix(M1, N1, M2, N2, x1_dt, x2_dt, D)
# print(D1.toarray())

# # Example usage for solve_advDif
# dt, T_end = 0.01, 1.0
# mu, sigma = [0, 0], 0.5 * np.eye(2)

# t, p = solve_advDif(D1, D2, D11, D22, x1, x2, dt, T_end, mu, sigma)
# print(p[:, :, -1])
