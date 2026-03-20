import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def step(state, sigma, tho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (tho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def runge_kutta_4(state, dt, sigma, tho, beta):
    k1 = step(state, sigma, tho, beta)
    k2 = step([state[i] + 0.5 * dt * k1[i] for i in range(3)], sigma, tho, beta)
    k3 = step([state[i] + 0.5 * dt * k2[i] for i in range(3)], sigma, tho, beta)
    k4 = step([state[i] + dt * k3[i] for i in range(3)], sigma, tho, beta)

    new_state = [state[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3)]
    return new_state

def simulate_lorenz(initial_state, dt, steps, sigma=10.0, tho=28.0, beta=8/3):
    states = [initial_state]
    current_state = initial_state

    for _ in range(steps):
        current_state = runge_kutta_4(current_state, dt, sigma, tho, beta)
        states.append(current_state)

    return states

def plot_trajectory(trajectory):
    x = [state[0] for state in trajectory]
    y = [state[1] for state in trajectory]
    z = [state[2] for state in trajectory]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_title("Lorenz Attractor")
    plt.show()


if __name__ == "__main__":
    initial_state = [1.0, 1.0, 1.0]
    dt = 0.01
    steps = 10000
    trajectory = simulate_lorenz(initial_state, dt, steps)

    # Print the final state after simulation
    print("Final state:", trajectory[-1])
    # Plot the trajectory of the Lorenz attractor
    plot_trajectory(trajectory)
