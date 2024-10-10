import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator, gmres

params = {"text.usetex": True, "font.size": 8, "font.family": "lmodern"}
plt.rcParams.update(params)

FIG_WIDTH = 11.7 / 2.54

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


def linrossler(t, u, upo, p):
    a, b, c = p
    X, Y, Z = upo(t)
    J = np.array([[0.0, -1, -1], [1, a, 0], [Z, 0, X - c]])
    return J @ u


def Φ(x, τ, p):
    tspan = (0, τ)
    sol = solve_ivp(
        lambda t, x: rossler(t, x, p),
        tspan,
        x,
        dense_output=True,
        atol=1e-10,
        rtol=1e-10,
        method="DOP853",
    )
    return sol.sol


def dΦ(x, X, τ, p):
    tspan = (0, τ)
    upo = Φ(X, τ, p)
    sol = solve_ivp(
        lambda t, x: linrossler(t, x, upo, p),
        tspan,
        x,
        atol=1e-10,
        rtol=1e-10,
        method="DOP853",
    )
    return sol.y[:, -1]


def linop(X, τ, p):
    mv = lambda x: dΦ(x, X, τ, p) - x
    A = LinearOperator((3, 3), matvec=mv)
    return A


def newton_upo(x, τ, p, tol=1e-6, maxiter=100):
    f = lambda x, τ: rossler(τ, x, p)
    g = lambda x, τ: rossler(τ, Φ(x, τ, p)(τ), p)

    for i in range(maxiter):
        r = Φ(x, τ, p)(τ) - x
        residual = np.linalg.norm(r) ** 2
        print(f"Residual at iteration {i}: {residual}")

        if residual < tol:
            break

        b = g(x, τ).reshape(-1, 1)
        c = f(x, τ).reshape(-1, 1)
        A = np.array([linop(x, τ, p) @ np.eye(3)[:, i] for i in range(3)]).T
        J = np.block([[A, b], [c.T, np.array([0])]])

        rhs = np.zeros(4)
        rhs[:3] = -r
        sol = np.linalg.pinv(J) @ rhs
        dx = sol[:3].flatten()
        dτ = sol[-1]

        print("Updated solution")
        x += dx
        τ += dτ
        print(x, τ)
        if τ < 0:
            break

    return x, τ


def upo1():
    a, b, cs = 0.1, 0.1, np.linspace(5.37, 5.38, 11)
    p = [a, b, cs[0]]
    τ0 = 6
    x0 = np.array([6.84746634, -19.4312737, 0.0110787168])
    x0 = Φ(x0, 100 * τ0, p)(100 * τ0)
    x0, τ0 = newton_upo(x0, τ0, p, maxiter=100, tol=1e-10)
    print(τ0)

    for c in cs:
        print(f" ----- c: {c}\n")
        p = [a, b, c]
        x0, τ0 = newton_upo(x0, τ0, p, maxiter=100, tol=1e-10)

        A = np.array([dΦ(np.eye(3)[:, i], x0, τ0, p) for i in range(3)])
        λ = np.linalg.eigvals(A)
        print(f"Eigenvalues: {λ}")
        if np.abs(λ).max() > 1.00001:
            print(f"BIFURCATION DETECTED at c = {c}")
            break

    return x0, τ0


def upo2():
    a, b, cs = 0.1, 0.1, np.linspace(7.77, 7.78, 11)
    p = [a, b, cs[0]]
    τ0 = 12
    x0 = np.array([6.84746634, -19.4312737, 0.0110787168])
    x0 = Φ(x0, 100 * τ0, p)(100 * τ0)

    for c in cs:
        print(f" ----- c: {c}\n")
        p = [a, b, c]
        x0, τ0 = newton_upo(x0, τ0, p, maxiter=100, tol=1e-10)

        A = np.array([dΦ(np.eye(3)[:, i], x0, τ0, p) for i in range(3)])
        λ = np.linalg.eigvals(A)
        print(f"Eigenvalues : {λ}")
        if np.abs(λ).max() > 1.00001:
            print(f"BIFURCATION DETECTED c = {c}")
            break

    return x0, τ0


def strange_attractor():
    p = [0.1, 0.1, 14]
    tspan = (0.0, 250.0)
    teval = np.linspace(*tspan, 50001)
    u0 = np.array([6.84746634, -19.4312737, 0.0110787168])

    # Solve the Rossler system ODE.
    sol = solve_ivp(
        lambda t, x: rossler(t, x, p), tspan, u0, t_eval=teval, dense_output=True
    )

    fig = plt.figure(figsize=(FIG_WIDTH / 2, FIG_WIDTH / 2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Print the range of the solution for verification.
    print(f"X range: {sol.y[0].max()} to {sol.y[0].min()}")
    print(f"Y range: {sol.y[1].max()} to {sol.y[1].min()}")
    print(f"Z range: {sol.y[2].max()} to {sol.y[2].min()}")

    ax.plot(sol.y[0], sol.y[1], sol.y[2], color="black", linewidth=0.1, alpha=0.75)
    ax.set_axis_off()
    ax.set(xlim=(-25, 25), ylim=(-25, 25), zlim=(0, 40))
    ax.view_init(30, -135)
    ax.dist = 7

    # Make the background transparent.
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.savefig("rossler_system.png", bbox_inches="tight", dpi=1200, transparent=False)
    return fig, ax


if __name__ == "__main__":
    
    #x1, τ1 = upo1()
    #x2, τ2 = upo2()
    #strange_attractor()
    
    a, b, c = 0.2, 0.2, 5.7

    p = [a, b, c]
    
    bfin = np.array([1,0,0])
    xpin = np.array([1,0,0])
    # output of nonlinear integration after period T
    res = Φ(bfin, 1, p)(1) - bfin
    print('r:')
    print(res)
    
    print('NL bfin:')
    print(rossler(0, bfin, p))
    print(' L r:')
    print(linrossler(0, -xpin, Φ(bfin, 1, p), p))
    
    po1 = Φ(bfin, 10, p)
    tv = np.linspace(0, 10, 100)
    plt.plot(tv, po1(tv))
    
    sys.exit()
    
    print('bfin:')
    print(bfin)
    print('bfout:')
    print(Φ(bfin, 1, p)(1))

    # output of nonlinear integration after period T
    res = Φ(bfin, 1, p)(1) - bfin
    print('r:')
    print(res)
    
    print('xpin:')
    print(-res)
    print('xpout:')
    print(dΦ(res, bfin, 1, p))
    
    sys.exit()
    
    f = lambda x, τ: rossler(τ, x, p)
    g = lambda x, τ: rossler(τ, Φ(x, τ, p)(τ), p)
    b = g(bfin, 1).reshape(-1, 1)
    c = f(bfin, 1).reshape(-1, 1)
    print("f'[X(T),T]:")
    print(b)
    print("f'[X(0),0]:")
    print(c)
    
    A = np.array([linop(bfin, 1, p) @ np.eye(3)[:, i] for i in range(3)]).T
    J = np.block([[A, b], [c.T, np.array([0])]])
    
    print("A @ -r:")
    print(A @ -res)
    
    res = np.append(res, 0.0)
    
    print("J @ -r_:")
    print(J @ -res)
    
    
    
    