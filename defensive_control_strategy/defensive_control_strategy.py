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
        striker = Object.Striker(Parameters.Striker.Radius, Parameters.Striker.Max_velocity)
        puck = Object.Puck(Parameters.Puck.Radius, Parameters.Puck.Max_velocity)

        self.play_board = Physical_Model.PlayBoard(Parameters.PlayBoard.L_x,
                                                   Parameters.PlayBoard.L_y,
                                                   Parameters.PlayBoard.D_x,
                                                   Parameters.PlayBoard.D_y,
                                                   striker,
                                                   puck,
                                                   Parameters.Simulation_ratio,
                                                   cuda=Parameters.cuda
                                                   )

    def run_simulation(self):

        distance_mean = 0
        for j in range(1000):
            for i in range(10000):
                self.play_board.new_game()

                if i % 10 == 0:
                    self.play_board.decision.zero_grad()

                distance = self.play_board.run_till_gameover()

                if i % 100 == 0:
                    print("{}st end distance is {:.3f}".format(i, float(distance_mean / 100)))
                    distance_mean = distance

                else:
                    distance_mean += distance

            for param_group in self.play_board.optimizer.param_groups:
                param_group['lr'] *= 0.5

            self.play_board.save_checkpoint(j)


if __name__ == "__main__":
    model = DefensiveModel()
    model.run_simulation()
