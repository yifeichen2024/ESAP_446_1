import numpy as np
import scipy
from scipy.special import factorial
from scipy import sparse
from scipy.linalg import lstsq, svd
import mpmath
from farray import apply_matrix, reshape_vector

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

    def dx_array(self, j):
        shape = (self.N, len(j))
        dx = np.zeros(shape)

        jmin = -np.min(j)
        jmax = np.max(j)

        values_padded = np.zeros(self.N + jmin + jmax)
        if jmin > 0:
            values_padded[:jmin] = self.values[-jmin:] - self.length
        if jmax > 0:
            values_padded[jmin:-jmax] = self.values
            values_padded[-jmax:] = self.length + self.values[:jmax]
        else:
            values_padded[jmin:] = self.values

        for i in range(self.N):
            dx[i, :] = values_padded[jmin+i+j] - values_padded[jmin+i]

        return dx
    
class Domain:
    def __init__(self, grids):
        self.dimension = len(grids)
        self.grids = grids
        shape = []
        for grid in self.grids:
            shape.append(grid.N)
            self.shape = shape
    
    def values(self):
        v = []
        for i, grid in enumerate(self.grids):
            grid_v = grid.values
            grid_v = reshape_vector(grid_v, self.dimension, i)
            v.append(grid_v)
        return v
    
class Difference:
    def __matmul__(self, other):
        return self.matrix @ other

    # Define a class that multiple the float with DifferenceUniformGrid
    def __mul__(self, other):
        return np.dot(other, self.matrix)
    
'''
Take derivatives of functions defined over a UniformPeriodicGrid
hw2_test1.py
'''
class DifferenceUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        if stencil_type == 'centered' and convergence_order % 2 != 0:
            raise ValueError("Centered finite difference has even convergence order")

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self._stencil_shape(stencil_type)
        self._make_stencil(grid)
        self._build_matrix(grid)

    def _stencil_shape(self, stencil_type):
        dof = self.derivative_order + self.convergence_order

        if stencil_type == 'centered':
            # cancellation if derivative order is even
            dof = dof - (1 - dof % 2)
            j = np.arange(dof) - dof//2

        self.dof = dof
        self.j = j

    def _make_stencil(self, grid):

        # assume constant grid spacing
        self.dx = grid.dx
        i = np.arange(self.dof)[:, None]
        j = self.j[None, :]
        S = 1/factorial(i)*(j*self.dx)**i

        b = np.zeros( self.dof )
        b[self.derivative_order] = 1.

        self.stencil = scipy.linalg.solve(S, b)

    def _build_matrix(self, grid):
        shape = [grid.N] * 2
        matrix = sparse.diags(self.stencil, self.j, shape=shape)
        matrix = matrix.tocsr()
        jmin = -np.min(self.j)
        if jmin > 0:
            for i in range(jmin):
                matrix[i,-jmin+i:] = self.stencil[:jmin-i]

        jmax = np.max(self.j)
        if jmax > 0:
            for i in range(jmax):
                matrix[-jmax+i,:i+1] = self.stencil[-i-1:]
        self.matrix = matrix
    
    
'''
Take derivatives of functions defined over a NonUniformPeriodicGrid
hw2_test2.py
'''
class DifferenceNonUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        if (derivative_order + convergence_order) % 2 == 0:
            raise ValueError("The derivative plus convergence order must be odd for centered finite difference")

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self._stencil_shape(stencil_type)
        self._make_stencil(grid)
        self._build_matrix(grid)

    def _stencil_shape(self, stencil_type):
        dof = self.derivative_order + self.convergence_order
        j = np.arange(dof) - dof//2
        self.dof = dof
        self.j = j

    def _make_stencil(self, grid):
        self.dx = grid.dx_array(self.j)

        i = np.arange(self.dof)[None, :, None]
        dx = self.dx[:, None, :]
        S = 1/factorial(i)*(dx)**i

        b = np.zeros(self.dof)
        b[self.derivative_order] = 1.

        self.stencil = np.linalg.solve(S, b)

    def _build_matrix(self, grid):
        shape = [grid.N] * 2
        diags = []
        for i, jj in enumerate(self.j):
            if jj < 0:
                s = slice(-jj, None, None)
            else:
                s = slice(None, None, None)
            diags.append(self.stencil[s, i])
        matrix = sparse.diags(diags, self.j, shape=shape)

        matrix = matrix.tocsr()
        jmin = -np.min(self.j)
        if jmin > 0:
            for i in range(jmin):
                matrix[i,-jmin+i:] = self.stencil[i, :jmin-i]

        jmax = np.max(self.j)
        if jmax > 0:
            for i in range(jmax):
                matrix[-jmax+i,:i+1] = self.stencil[-jmax+i, -i-1:]

        self.matrix = matrix


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


class CenteredFiniteDifference6(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-3, -2, -1, 0, 1, 2, 3]
        diags = np.array([-1, 9, -45, 0, 45, -9, 1]) / (60 * h)
        matrix = sparse.diags(diags, offsets=j, shape=[N, N])
        matrix = matrix.tocsr()

        # 处理边界条件
        matrix[-3, 0] = -1/(60*h)
        matrix[-2, 0] = 9/(60*h)
        matrix[-1, 0] = -45/(60*h)
        matrix[-1, 1] = 45/(60*h)
        matrix[-1, 2] = -9/(60*h)
        matrix[-1, 3] = 1/(60*h)

        matrix[0, -3] = 1/(60*h)
        matrix[0, -2] = -9/(60*h)
        matrix[0, -1] = 45/(60*h)
        self.matrix = matrix

class CenteredFiniteFourthDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-2, -1, 0, 1, 2]
        diags = np.array([1, -4, 6, -4, 1]) / (h**4)
        matrix = sparse.diags(diags, offsets=j, shape=[N, N])
        matrix = matrix.tocsr()

        # 处理边界条件
        matrix[-2, 0] = 1/(h**4)
        matrix[-1, 0] = -4/(h**4)
        matrix[-1, 1] = 6/(h**4)
        matrix[-1, 2] = -4/(h**4)
        matrix[-1, 3] = 1/(h**4)

        matrix[0, -2] = 1/(h**4)
        matrix[0, -1] = -4/(h**4)
        matrix[1, -1] = 6/(h**4)
        self.matrix = matrix

class HigherOrderCenteredDifference(Difference):
    def __init__(self, grid, derivative_order, accuracy_order):
        h = grid.dx
        N = grid.N

        if derivative_order == 1 and accuracy_order == 4:
            # Fourth-order accurate first derivative
            j = [-2, -1, 0, 1, 2]
            diags = np.array([1/(12*h), -8/(12*h), 0, 8/(12*h), -1/(12*h)])
        elif derivative_order == 2 and accuracy_order == 4:
            # Fourth-order accurate second derivative
            j = [-2, -1, 0, 1, 2]
            diags = np.array([-1/(12*h**2), 16/(12*h**2), -30/(12*h**2), 16/(12*h**2), -1/(12*h**2)])
        elif derivative_order == 1 and accuracy_order == 6:
            # Sixth-order accurate first derivative
            j = [-3, -2, -1, 0, 1, 2, 3]
            diags = np.array([-1/(60*h), 9/(60*h), -45/(60*h), 0, 45/(60*h), -9/(60*h), 1/(60*h)])
        elif derivative_order == 2 and accuracy_order == 6:
            # Sixth-order accurate second derivative
            j = [-3, -2, -1, 0, 1, 2, 3]
            diags = np.array([2/(180*h**2), -27/(180*h**2), 270/(180*h**2), -490/(180*h**2), 270/(180*h**2), -27/(180*h**2), 2/(180*h**2)])
        elif derivative_order == 1 and accuracy_order == 8:
            # Eighth-order accurate first derivative (new)
            j = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
            diags = np.array([1/(280*h), -4/(105*h), 1/(5*h), -4/(5*h), 0, 4/(5*h), -1/(5*h), 4/(105*h), -1/(280*h)])
        elif derivative_order == 2 and accuracy_order == 8:
            # Eighth-order accurate second derivative (new)
            j = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
            diags = np.array([-1/(560*h**2), 8/(315*h**2), -1/(5*h**2), 8/(5*h**2), -205/(72*h**2), 8/(5*h**2), -1/(5*h**2), 8/(315*h**2), -1/(560*h**2)])
        else:
            raise ValueError("The specified derivative order and accuracy order are not implemented.")

        matrix = sparse.diags(diags, offsets=j, shape=[N, N])
        matrix = matrix.tolil()

        # Handling periodic boundary conditions
        for idx, offset in enumerate(j):
            if offset < 0:
                matrix[:abs(offset), offset + N:] = diags[idx]
                matrix[-abs(offset):, :abs(offset)] = diags[idx]
            elif offset > 0:
                matrix[-offset:, :offset] = diags[idx]
                matrix[:offset, -offset:] = diags[idx]

        self.matrix = matrix.tocsr()