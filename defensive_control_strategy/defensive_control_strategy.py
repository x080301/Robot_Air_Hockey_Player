# Monte Carlo Policy Gradient
#
# Initialisiere Strategieparameter zufällig
# DO:
# Generiere Episoden
# berechne Gradient
# aktualisiere Strategieparameter
# WHILE Iteration < N

from ..DL_model import FCNet
from ..physical_model import Object, Physical_Model


class Parameters:
    class Striker:
        Radius = 8

    class Puck:
        Radius = 2

    class PlayBoard:
        L_x = 50
        L_y = 100
        D_x = 10
        D_y = 15


def init_simulation():
    striker = Object.Striker(Parameters.Striker.Radius)
    puck = Object.Puck(Parameters.Puck.Radius)
    play_board = Physical_Model.PlayBoard(Parameters.PlayBoard.L_x,
                                          Parameters.PlayBoard.L_y,
                                          Parameters.PlayBoard.D_x,
                                          Parameters.PlayBoard.D_y)


if __name__ == "__main__":
    init_simulation()
