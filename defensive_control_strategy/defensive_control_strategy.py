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
from auxiliary_function.auxiliary_function import plot_result


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

    def test(self, check_point):
        self.play_board.restore_checkpoint(check_point)

        Hit_num = 0
        for i in range(1000):

            self.play_board.new_game()
            distance, _ = self.play_board.run_test()
            if distance <= (Parameters.Puck.Radius + Parameters.Striker.Radius):
                Hit_num += 1
            #print(distance)
        Hit_rat = Hit_num / 1000
        print(Hit_rat)

        self.play_board.new_game()
        distance, state_stack = self.play_board.run_test()
        Px = state_stack[:, 0].cpu().numpy()
        Py = state_stack[:, 1].cpu().numpy()

        Sx = state_stack[:, 4].cpu().numpy()
        Sy = state_stack[:, 5].cpu().numpy()
        print(distance)

        plot_result(Px, Py, Sx, Sy)

    def fine_simulation(self):

        distance_mean = 0
        K = 100

        for i in range(100000):
            self.play_board.new_game()

            distance = self.play_board.run_till_gameover()

            self.play_board.optimizer.step()

            if i % K == 0:
                self.play_board.decision.zero_grad()
                print("{}st end distance is {:.3f}".format(i, float(distance_mean / K)))
                self.play_board.save_checkpoint(i, distance_mean / K)
                distance_mean = distance
            else:
                distance_mean += distance

    def run_simulation(self, fine_rate=None, check_point=None):
        distance_mean = 0
        K = 50
        if fine_rate is not None:
            self.play_board.restore_checkpoint(check_point)
            for param_group in self.play_board.optimizer.param_groups:
                param_group['lr'] *= fine_rate

        for i in range(10000):
            self.play_board.new_game()

            distance = self.play_board.run_till_gameover()

            # self.play_board.optimizer.step()
            if i % K == 0:

                self.play_board.optimizer.step()
                self.play_board.decision.zero_grad()
                print("{}st end distance is {:.3f}".format(i, float(distance_mean / K)))
                self.play_board.save_checkpoint(i, distance_mean / K)
                distance_mean = distance
            else:
                distance_mean += distance

            """if i % 1000 == 0:
                for param_group in self.play_board.optimizer.param_groups:
                    param_group['lr'] *= 0.5"""


if __name__ == "__main__":
    model = DefensiveModel()
    #model.run_simulation(fine_rate=0.01, check_point='checkpoints/checkpoint_0400_7.422.ckp')
    model.test('checkpoints/checkpoint_1300_5.088.ckp')
