# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def simulate_underdamped(m, k, zeta, x0, v0, t):
    omega0 = np.sqrt(k/m)
    omega_d = omega0*np.sqrt(1 - zeta**2)
    A = x0
    B = (zeta*omega0*x0 + v0)/omega_d
    x = np.exp(-zeta*omega0*t)*(A*np.cos(omega_d*t) + B*np.sin(omega_d*t))
    v = np.exp(-zeta*omega0*t)*(
        -zeta*omega0*(A*np.cos(omega_d*t) + B*np.sin(omega_d*t))
        - A*omega_d*np.sin(omega_d*t) + B*omega_d*np.cos(omega_d*t)
    )
    return x, v

def simulate_critical(m, k, x0, v0, t):
    omega0 = np.sqrt(k/m)
    C1 = x0
    C2 = omega0*x0  # v0 = 0
    x = (C1 + C2*t)*np.exp(-omega0*t)
    v = (C2 - omega0*(C1 + C2*t))*np.exp(-omega0*t)
    return x, v

def simulate_overdamped(m, k, zeta, x0, v0, t):
    omega0 = np.sqrt(k/m)
    s = np.sqrt(zeta**2 - 1)
    r1 = -omega0*(zeta - s)
    r2 = -omega0*(zeta + s)
    A = np.array([[1, 1], [r1, r2]])
    b = np.array([x0, v0])
    C1, C2 = np.linalg.solve(A, b)
    x = C1*np.exp(r1*t) + C2*np.exp(r2*t)
    v = C1*r1*np.exp(r1*t) + C2*r2*np.exp(r2*t)
    return x, v

def main():
    m = 1.0
    k = 1.0
    x0 = 1.0
    v0 = 0.0
    t = np.linspace(0, 20, 2000)

    regimes = {
        'Underdamped (ζ=0.5)': simulate_underdamped(m, k, 0.5, x0, v0, t),
        'Critical (ζ=1.0)': simulate_critical(m, k, x0, v0, t),
        'Overdamped (ζ=1.5)': simulate_overdamped(m, k, 1.5, x0, v0, t)
    }

    # Plot displacement vs time
    plt.figure()
    for label, (x, _) in regimes.items():
        plt.plot(t, x, label=label)
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.title('Displacement vs Time for Different Damping Regimes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('displacement_vs_time.png')

    # Plot phase space trajectories
    plt.figure()
    for label, (x, v) in regimes.items():
        plt.plot(x, v, label=label)
    plt.xlabel('Displacement')
    plt.ylabel('Velocity')
    plt.title('Phase Space Trajectories')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('phase_space.png')

    # Primary numeric answer: natural frequency ω0
    omega0 = np.sqrt(k/m)
    print('Answer:', omega0)

if __name__ == '__main__':
    main()

