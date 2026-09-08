import matplotlib.pyplot as plt

from chaosdyn import Lorenz
from chaosdyn.solvers import solve


def main():
    system = Lorenz()

    times, trajectory = solve(
        system,
        initial_state=[1.0, 1.0, 1.0],
        t_span=(0.0, 40.0),
        dt=0.01,
    )

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=0.5)

    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


if __name__ == "__main__":
    main()