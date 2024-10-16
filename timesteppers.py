import numpy as np
from scipy import sparse
class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1
        
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, u, f):
        super().__init__()
        self.u = u
        self.f = f


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.u + dt*self.f(self.u)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.f(self.u)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.f(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.f(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, f1, f2):
        super().__init__()
        self.u = u
        self.f1 = f1
        self.f2 = f2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(ExplicitTimestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b

    def _step(self, dt):
        k = [np.zeros_like(self.u) for _ in range(self.stages)]
        for i in range(self.stages):
            u_temp = np.copy(self.u)
            for j in range(i):
                u_temp += dt * self.a[i, j] * k[j]
            k[i] = self.f(u_temp)
        return self.u + dt * sum(self.b[i] * k[i] for i in range(self.stages))


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt
        self.previous_fs = []

    def _step(self, dt):
        if len(self.previous_fs) < self.steps - 1:
            # Use Forward Euler for the first few steps
            self.previous_fs.append(self.f(self.u))
            return self.u + dt * self.f(self.u)
        else:
            # Use Adams-Bashforth method
            self.previous_fs.append(self.f(self.u))
            if len(self.previous_fs) > self.steps:
                self.previous_fs.pop(0)
            coefficients = self._get_coefficients()
            return self.u + dt * sum(coeff * f_val for coeff, f_val in zip(coefficients, reversed(self.previous_fs)))

    def _get_coefficients(self):
        # Coefficients for Adams-Bashforth methods of different orders
        if self.steps == 1:
            return [1]
        elif self.steps == 2:
            return [3/2, -1/2]
        elif self.steps == 3:
            return [23/12, -16/12, 5/12]
        elif self.steps == 4:
            return [55/24, -59/24, 37/24, -9/24]
        else:
            raise ValueError("Adams-Bashforth method only implemented for up to 4 steps.")
