import matplotlib.pyplot as plt
from physical_model.Physical_Model import Parameters


def plot_result(Px_axis_data, Py_axis_data, Sx_axis_data, Sy_axis_data):
    fig, ax = plt.subplots()

    plt.xlim((0, Parameters.PlayBoard.L_x))
    plt.ylim((0, Parameters.PlayBoard.L_y))

    for i in range(Py_axis_data.shape[0]):
        circle = plt.Circle((Px_axis_data[i], Py_axis_data[i]), 8, fill=False)

        ax.add_artist(circle)

    for i in range(Sy_axis_data.shape[0]):
        circle = plt.Circle((Sx_axis_data[i], Sy_axis_data[i]), 2, fill=False)

        ax.add_artist(circle)

    plt.show()
