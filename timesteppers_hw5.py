import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque

class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.dt = dt
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


class ImplicitTimestepper(Timestepper):

    def __init__(self, u, L):
        super().__init__()
        self.u = u
        self.L = L
        N = len(u)
        self.I = sparse.eye(N, N)


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
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = f1
        self.f2 = f2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(ExplicitTimestepper):

    def __init__(self, u, f, stages, a, b):
        pass

    def _step(self, dt):
        pass


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, u, f, steps, dt):
        pass

    def _step(self, dt):
        pass


class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.u)


class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.L.matrix
            self.RHS = self.I + dt/2*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.RHS @ self.u)


class BackwardDifferentiationFormula(ImplicitTimestepper):

    def __init__(self, u, L, steps):
        pass

    def _step(self, dt):
        pass


class StateVector:
    
    def __init__(self, variables):
        var0 = variables[0]
        self.N = len(var0)
        size = self.N*len(variables)
        self.data = np.zeros(size)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[i*self.N:(i+1)*self.N], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[i*self.N:(i+1)*self.N])


class IMEXTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__()
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F

    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.dt = dt
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        
        RHS = self.M @ self.X.data + dt*self.F(self.X)
        return self.LU.solve(RHS)


class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt or self.iter == 1:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)


class BDFExtrapolate(IMEXTimestepper):

    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        pass

    def _step(self, dt):
        pass

# class BDFExtrapolate(IMEXTimestepper):
    
#     def __init__(self, eq_set, steps):
#         super().__init__(eq_set)
#         self.steps = steps
#         self.previous_X = deque(maxlen=steps)
#         self.previous_F = deque(maxlen=steps)
#         self.coefficients_a = None
#         self.coefficients_b = None

#     def _compute_coefficients(self):
#         # Compute coefficients for BDF and extrapolation using factorials
#         a = np.zeros(self.steps + 1)
#         b = np.zeros(self.steps)
        
#         # Calculate BDF coefficients a_0, a_1, ..., a_s
#         for i in range(self.steps + 1):
#             product = 1
#             for j in range(self.steps + 1):
#                 if j != i:
#                     product *= (self.steps - j) / (i - j)
#             a[i] = product
        
#         # Calculate extrapolation coefficients b_1, b_2, ..., b_s
#         for i in range(1, self.steps + 1):
#             b[i - 1] = (-1) ** (i + 1) * factorial(self.steps) / (factorial(i) * factorial(self.steps - i))
        
#         self.coefficients_a = a
#         self.coefficients_b = b

#     def _step(self, dt):
#         if self.coefficients_a is None or self.coefficients_b is None:
#             self._compute_coefficients()
        
#         if len(self.previous_X) < self.steps:
#             # Use a lower-order method for the first few steps
#             LHS = self.M + dt * self.L
#             LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
#             RHS = self.M @ self.X.data + dt * self.F(self.X)
#             result = LU.solve(RHS)
#         else:
#             # Use full BDFExtrapolate scheme
#             # Compute the linear combination of previous values using coefficients
#             dX_dt_approx = -self.coefficients_a[0] * self.X.data
#             for i in range(1, self.steps + 1):
#                 dX_dt_approx += self.coefficients_a[i] * self.previous_X[-i]
            
#             # Compute the extrapolation of F(X)
#             F_approx = np.zeros_like(self.X.data)
#             for i in range(1, self.steps + 1):
#                 F_approx += self.coefficients_b[i - 1] * self.previous_F[-i]
            
#             # Construct LHS and RHS for solving
#             LHS = self.M
#             RHS = self.M @ self.X.data + dt * (self.L @ dX_dt_approx + F_approx)
            
#             # Solve the linear system
#             LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
#             result = LU.solve(RHS)
        
#         # Store current values for future use
#         self.previous_X.append(np.copy(self.X.data))
#         self.previous_F.append(self.F(self.X))
        
#         return result