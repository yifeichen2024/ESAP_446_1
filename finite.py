import numpy as np
from scipy.special import factorial
from scipy import sparse
from scipy.linalg import lstsq, svd
import mpmath

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class Difference:
    # def __init__(self):
    #     self.matrix = None
    def __matmul__(self, other):
        return self.matrix @ other


'''
Take derivatives of functions defined over a UniformPeriodicGrid
hw2_test1.py
'''
class DifferenceUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self.grid = grid 
        self.matrix = self._construct_derivative_matrix()
        
    def _construct_derivative_matrix(self):
        N = self.grid.N
        dx = self.grid.dx
        stencil = self._compute_stencil()
        stencil_size = len(stencil)
        offsets = np.arange(-(stencil_size // 2), stencil_size // 2 + 1)

        data = np.tile(stencil, (N, 1))
        rows = np.arange(N).repeat(stencil_size)
        cols = (rows.reshape(-1, stencil_size) + offsets) % N

        # Create sparse matrix using the computed stencil
        derivative_matrix = sparse.coo_matrix((data.ravel(), (rows, cols.ravel())), shape=(N, N))
        return (derivative_matrix / (dx ** self.derivative_order)).tocsr()
    
    def _compute_stencil(self):
        # Calculate the stencil for the specified derivative and convergence order
        order = self.derivative_order
        convergence = self.convergence_order
        stencil = []

        if self.stencil_type == 'centered':
            # Using finite difference coefficients for a centered stencil
            points = np.arange(-convergence, convergence + 1)
            A = np.vander(points, increasing=True).T
            b = np.zeros(len(points))
            b[order] = factorial(order)
            stencil = np.linalg.solve(A, b)
        else:
            raise NotImplementedError("Only centered stencils are implemented.")

        return stencil
    
'''
Take derivatives of functions defined over a NonUniformPeriodicGrid
hw2_test2.py
'''
class DifferenceNonUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        super().__init__()
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self.grid = grid
        # np.random.seed(42)
        self.matrix = self._construct_derivative_matrix()

    def _construct_derivative_matrix(self):
        N = self.grid.N
        values = self.grid.values
        stencil_size = min(2 * self.convergence_order + 3, N)  # Increase stencil size for better accuracy
        data = []
        rows = []
        cols = []

        for i in range(N):
            stencil_indices = self._get_stencil_indices(i, N, stencil_size)
            stencil_points = values[stencil_indices] - values[i]
            A = np.vander(stencil_points, increasing=True).T
            b = np.zeros(stencil_size)
            b[self.derivative_order] = factorial(self.derivative_order)

            # Use high precision solver for better numerical stability
            with mpmath.workdps(100):
                A_mp = mpmath.matrix(A)
                b_mp = mpmath.matrix(b)
                stencil_mp = mpmath.lu_solve(A_mp, b_mp)
            stencil = np.array(stencil_mp, dtype=np.float64)

            # # Use SVD for better numerical stability
            # U, s, Vt = svd(A, full_matrices=False)
            # # c = np.dot(U.T, b) / s
            # c = np.dot(U.T, b) / (s + 1e-12)  # Adding a higher value to s to improve stability
            # stencil = np.dot(Vt.T, c)

            data.extend(stencil)
            rows.extend([i] * stencil_size)
            cols.extend(stencil_indices)

            # Create sparse matrix using the computed stencil
        derivative_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(N, N))
        return derivative_matrix.tocsr()

    def _get_stencil_indices(self, i, N, stencil_size):
        half_size = stencil_size // 2
        if i < half_size:
            # Use asymmetric stencil near the start
            return np.arange(0, stencil_size)
        elif i >= N - half_size:
            # Use asymmetric stencil near the end
            return np.arange(N - stencil_size, N)
        else:
            # Use symmetric stencil in the middle
            return np.arange(i - half_size, i + half_size + 1)

    def _compute_weights(self, stencil_points, order):
        A = np.vander(stencil_points, increasing=True).T
        b = np.zeros(len(stencil_points))
        b[order] = factorial(order)
        weights, _, _, _ = lstsq(A, b)  # Use least squares for better numerical accuracy
        return weights
    

class ForwardFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [0, 1]
        diags = np.array([-1/h, 1/h])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h
        self.matrix = matrix


class CenteredFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([-1/(2*h), 0, 1/(2*h)])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/(2*h)
        matrix[0, -1] = -1/(2*h)
        self.matrix = matrix


class CenteredFiniteSecondDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([1/h**2, -2/h**2, 1/h**2])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h**2
        matrix[0, -1] = 1/h**2
        self.matrix = matrix


class CenteredFiniteDifference4(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-2, -1, 0, 1, 2]
        diags = np.array([1, -8, 0, 8, -1])/(12*h)
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-2, 0] = -1/(12*h)
        matrix[-1, 0] = 8/(12*h)
        matrix[-1, 1] = -1/(12*h)

        matrix[0, -2] = 1/(12*h)
        matrix[0, -1] = -8/(12*h)
        matrix[1, -1] = 1/(12*h)
        self.matrix = matrix

