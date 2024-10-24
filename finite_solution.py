import numpy as np
from scipy.special import factorial
from scipy import sparse
import scipy

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


class Difference:

    def __matmul__(self, other):
        return self.matrix @ other


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

