import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 1. 定义计算参数: 
# Linear oscillator
x1_dt = np.array([[1, 0, 1]])  # f1(x1, x2) = x2
x2_dt = np.array([[-0.2, 0, 1],
                  [-1, 1, 0]])  # f2(x1, x2) = -0.2*x2 - x1
D = np.array([0, 0.2])
dt = 0.01
T_end = 25
M1, M2 = 10, 10
N1, N2 = 200, 200
mu = np.array([5, 5])
sigma = (1 / 9) * np.eye(2)

t = np.arange(0, T_end + dt, dt)

dx1 = 2 * M1 / (N1 - 1)
dx2 = 2 * M2 / (N2 - 1)
x1 = np.linspace(-M1, M1, N1)
x2 = np.linspace(-M2, M2, N2)
X1, X2 = np.meshgrid(x1, x2, indexing='ij')

pos = np.dstack((X1, X2))
p0 = multivariate_normal(mu, sigma).pdf(pos)

# 2. 定义漂移和扩散项
def compute_drift_terms(X1, X2, x1_dt, x2_dt):
    f1 = np.zeros_like(X1)
    f2 = np.zeros_like(X2)
    for coeff in x1_dt:
        a, b, c = coeff
        f1 += a * (X1 ** b) * (X2 ** c)
    for coeff in x2_dt:
        a, b, c = coeff
        f2 += a * (X1 ** b) * (X2 ** c)
    return f1, f2

f1, f2 = compute_drift_terms(X1, X2, x1_dt, x2_dt)

# 3. 构建离散化矩阵
def build_tridiagonal_matrix(N, lower, diag, upper):
    diagonals = [lower, diag, upper]
    offsets = [-1, 0, 1]
    return diags(diagonals, offsets, shape=(N, N))

D1 = D[0]
a_D1 = (D1 / dx1 ** 2) * np.ones(N1 - 1)
b_D1 = (-2 * D1 / dx1 ** 2) * np.ones(N1)
c_D1 = (D1 / dx1 ** 2) * np.ones(N1 - 1)
D1_matrix = build_tridiagonal_matrix(N1, a_D1, b_D1, c_D1).tocsr()

D2 = D[1]
a_D2 = (D2 / dx2 ** 2) * np.ones(N2 - 1)
b_D2 = (-2 * D2 / dx2 ** 2) * np.ones(N2)
c_D2 = (D2 / dx2 ** 2) * np.ones(N2 - 1)
D2_matrix = build_tridiagonal_matrix(N2, a_D2, b_D2, c_D2).tocsr()

# 4. 定义辅助函数
def apply_boundary_conditions(p):
    p[0, :] = 0
    p[-1, :] = 0
    p[:, 0] = 0
    p[:, -1] = 0
    return p

def normalize_probability(p, dx1, dx2):
    total_prob = np.sum(p) * dx1 * dx2
    return p / total_prob

# 5. 实现 ADI 方法
p = p0.copy()
N_steps = len(t)
p_record = []

for step in range(N_steps):
    if step % 100 == 0:
        print(f"Time step {step}/{N_steps}")
    
    # 第一步
    rhs = np.zeros_like(p)
    for j in range(N2):
        drift_term_x1 = - (f1[:, j] * (np.roll(p[:, j], -1) - np.roll(p[:, j], 1)) / (2 * dx1))
        rhs[:, j] = p[:, j] + (dt / 2) * (drift_term_x1)
        rhs[:, j] += (dt / 2) * D2 * (np.roll(p[:, j], -1) - 2 * p[:, j] + np.roll(p[:, j], 1)) / dx2 ** 2
    rhs = apply_boundary_conditions(rhs)
    for j in range(N2):
        A = csr_matrix(np.eye(N1) - (dt / 2) * D1_matrix.toarray())
        p[:, j] = spsolve(A, rhs[:, j])

    # 第二步
    rhs = np.zeros_like(p)
    for i in range(N1):
        drift_term_x2 = - (f2[i, :] * (np.roll(p[i, :], -1) - np.roll(p[i, :], 1)) / (2 * dx2))
        rhs[i, :] = p[i, :] + (dt / 2) * (drift_term_x2)
        rhs[i, :] += (dt / 2) * D1 * (np.roll(p[i, :], -1) - 2 * p[i, :] + np.roll(p[i, :], 1)) / dx1 ** 2
    rhs = apply_boundary_conditions(rhs)
    for i in range(N1):
        A = csr_matrix(np.eye(N2) - (dt / 2) * D2_matrix.toarray())
        p[i, :] = spsolve(A, rhs[i, :])
    p = apply_boundary_conditions(p)
    p = normalize_probability(p, dx1, dx2)
    if step % 50 == 0:
        p_record.append(p.copy())

# 计算最大概率密度值
max_p = max([np.max(p) for p in p_record])
X1_reduced = X1[::2, ::2]
X2_reduced = X2[::2, ::2]
# 创建三维动画
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def init():
    ax.clear()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Probability Density')
    ax.set_zlim(0, max_p)
    return []

def update(frame):
    ax.clear()
    p_reduced = p_record[frame][::2, ::2]
    surf = ax.plot_surface(X1_reduced, X2_reduced, p_reduced, cmap='viridis')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Probability Density')
    ax.set_title(f'Time = {frame * dt * 50:.2f}')
    ax.set_zlim(0, max_p)
    # 可选：调整视角
    ax.view_init(elev=30, azim=frame * 4)
    return [surf]

anim = FuncAnimation(fig, update, frames=len(p_record), init_func=init, blit=False)

# 保存动画（可选）
# anim.save('probability_density_3d_animation.mp4', writer='ffmpeg', fps=5)

plt.show()
# 6. 可视化结果
# fig, ax = plt.subplots(figsize=(8, 6))
# contour = ax.contourf(X1, X2, p_record[0], levels=50, cmap='viridis')
# fig.colorbar(contour, ax=ax)
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')

# def update(frame):
#     ax.clear()
#     contour = ax.contourf(X1, X2, p_record[frame], levels=50, cmap='viridis')
#     ax.set_xlabel('$x_1$')
#     ax.set_ylabel('$x_2$')
#     ax.set_title(f'Time = {frame * dt * 50:.2f}')
#     return contour.collections

# anim = FuncAnimation(fig, update, frames=len(p_record), interval=200)
# plt.show()
