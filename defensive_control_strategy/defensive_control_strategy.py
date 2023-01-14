# Monte Carlo Policy Gradient
#
# Initialisiere Strategieparameter zufällig
# DO:
# Generiere Episoden
# berechne Gradient
# aktualisiere Strategieparameter
# WHILE Iteration < N


from physical_model import Object, Physical_Model
from physical_model.Parameters import Parameters


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

            print(1)
            for j in range(1000):
                print(2)
                distance = self.play_board.run_till_strike()

                print("end distance is {:.3f}".format(float(distance)))

                if j == 20:
                    self.play_board.save_checkpoint(i)


if __name__ == "__main__":
    model = DefensiveModel()
    model.run_simulation()
