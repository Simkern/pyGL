import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator, gmres

params = {"text.usetex": True, "font.size": 8, "font.family": "lmodern"}
plt.rcParams.update(params)

FIG_WIDTH = 11.7 / 2.54

plt.close('all')

'''
def newton_krylov(
    x: np.ndarray, tau: float, max_iter: int = 100, tol: float = 1e-8
) -> tuple[np.ndarray, float]:
    for iteration in range(max_iter):
        # Compute the residual vector.
        residual_vector = F(x, tau)
        residual_norm = np.linalg.norm(residual_vector)
        print(f"Iteration {iteration}: Residual norm = {residual_norm}")

        # Check for convergence.
        if residual_norm < tol:
            print("Convergence achieved.")
            break

        # Construct the Jacobian operator.
        jacobian_operator = linop(x, tau)

        # Solve the linear system using GMRES.
        delta_x, info = gmres(jacobian_operator, -residual_vector)

        if info != 0:
            print(f"GMRES did not converge at iteration {iteration}, info = {info}")
            break

        # Update the solution.
        x += delta_x
        print(f"Updated solution: x = {x}")

    return x, residual_norm
'''

def rossler(t, u, p):
    a, b, c = p
    x, y, z = u
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz])

def linrossler(t, u, Xb, p):
    a, b, c = p
    X, Y, Z = Xb
    J = np.array([[0.0, -1.0, -1.0], [1.0, a, 0.0], [Z, 0.0, X - c]])
    return J @ u

def F(x, p):
    return rossler(0, x, p)


def dF(x, Xb, p):
    return linrossler(0, x, Xb, p)


def linop(Xb, p):
    mv = lambda x: dF(x, Xb, p)
    A = LinearOperator((3, 3), matvec=mv)
    return A


def newton(x, p, tol=1e-6, maxiter=100):

    for i in range(maxiter):
        r = F(x, p)
        residual = np.linalg.norm(r)**2
        print(f"Residual at iteration {i}: {residual}")

        if residual < tol:
            break

        J = np.array([linop(x, p) @ np.eye(3)[:, i] for i in range(3)]).T

        dx = np.linalg.pinv(J) @ -r

        print("Updated solution")
        x += dx
        print(x)

    return x

def fp(p, x0):
    return newton(x0, p, maxiter=100, tol=1e-10)


def strange_attractor(p):
    tspan = (0.0, 250.0)
    teval = np.linspace(*tspan, 50001)
    u0 = np.array([1, -1, 0])

    # Solve the Rossler system ODE.
    sol = solve_ivp(
        lambda t, x: rossler(t, x, p), tspan, u0, t_eval=teval, dense_output=True
    )

    fig = plt.figure(figsize=(FIG_WIDTH / 2, FIG_WIDTH / 2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Print the range of the solution for verification.
    print(f"X range: {sol.y[0].min()} to {sol.y[0].max()}")
    print(f"Y range: {sol.y[1].min()} to {sol.y[1].max()}")
    print(f"Z range: {sol.y[2].min()} to {sol.y[2].max()}")

    ax.plot(sol.y[0], sol.y[1], sol.y[2], color="black", linewidth=0.1, alpha=0.75)
    ax.set_axis_off()
    ax.set(xlim=(-25, 25), ylim=(-25, 25), zlim=(0, 40))
    ax.view_init(30, -135)
    ax.dist = 7

    # Make the background transparent.
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    #plt.savefig("rossler_system.png", bbox_inches="tight", dpi=1200, transparent=False)
    return fig, ax

def fp_analytical(p):
    a, b, c = p
    d = np.sqrt(c**2 - 4*a*b)
    fp1 = np.array([(c-d)/2, (-c+d)/(2*a), (c-d)/(2*a)])
    fp2 = np.array([(c+d)/2, (-c-d)/(2*a), (c+d)/(2*a)])
    return fp1, fp2


if __name__ == "__main__":
    p = [0.2, 0.2, 5.7]
    fig, ax = strange_attractor(p)
    fp1, fp2 = fp_analytical(p)
    print(fp1)
    print(fp2)
    x, y, z = fp1
    ax.scatter(x, y, z, c='red', s=20)
    x, y, z = fp2
    ax.scatter(x, y, z, c='blue', s=20)
    ax.scatter(0,0,0, c='black', s=10, marker='x')
    x0 = np.array([0.0, 0.0, 0.0])
    x1 = fp(p, x0)
    x, y, z = x1
    ax.scatter(x, y, z, c='black', s=50, marker='x')
    x0 = np.array([5.0, -10.0, 10.0])
    x1 = fp(p, x0)
    x, y, z = x1
    ax.scatter(x, y, z, c='black', s=50, marker='x')