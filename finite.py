import numpy as np
from scipy.special import factorial
from scipy import sparse
from scipy.linalg import lstsq, solve

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
hw2_test1
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
hw2_test1
'''
class DifferenceNonUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        super().__init__()
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self.grid = grid
        np.random.seed(42)
        self.matrix = self._construct_derivative_matrix()

    def _construct_derivative_matrix(self):
        N = self.grid.N
        values = self.grid.values
        stencil_size = 2 * self.convergence_order + 1
        data = []
        rows = []
        cols = []

        for i in range(N):
            stencil_indices = self._get_stencil_indices(i, N, stencil_size)
            stencil_points = values[stencil_indices] - values[i]
            A = np.vander(stencil_points, increasing=True).T
            b = np.zeros(stencil_size)
            b[self.derivative_order] = factorial(self.derivative_order)
            stencil, _, _, _ = lstsq(A, b)  # Use least squares for better numerical accuracy

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
    