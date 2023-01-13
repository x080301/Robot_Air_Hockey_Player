# Monte Carlo Policy Gradient
#
# Initialisiere Strategieparameter zufällig
# DO:
# Generiere Episoden
# berechne Gradient
# aktualisiere Strategieparameter
# WHILE Iteration < N


from physical_model import Object, Physical_Model


class Parameters:
    Simulation_ratio = 0.02  # Hz-1
    sigma = 0.5  # gauss distribution of strike velocity

    class Striker:
        Radius = 8.0
        Max_velocity = 25  # per s, one dimension

    class Puck:
        Radius = 2.0
        Max_velocity = 50  # per s, one dimension

    class PlayBoard:
        L_x = 50.0
        L_y = 100.0
        D_x = 10.0  # minimum distance between Striker and longitudinal edge
        D_y = 15.0  # minimum distance between Striker and Horizontal edge


class DefensiveModel:
    def __init__(self):
        striker = Object.Striker(Parameters.Striker.Radius,
                                 Parameters.Striker.Max_velocity)
        puck = Object.Puck(Parameters.Puck.Radius,
                           Parameters.Puck.Max_velocity)
        self.play_board = Physical_Model.PlayBoard(Parameters.PlayBoard.L_x,
                                                   Parameters.PlayBoard.L_y,
                                                   Parameters.PlayBoard.D_x,
                                                   Parameters.PlayBoard.D_y,
                                                   striker,
                                                   puck,
                                                   Parameters.Simulation_ratio
                                                   )

    def run_simulation(self):

        for i in range(100):

            self.play_board.new_game()
            self.play_board.update_decision_model()




if __name__ == "__main__":
    model = DefensiveModel()
    model.run_simulation()
