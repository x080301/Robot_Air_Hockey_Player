from ..DL_model import FCNet
import torch


class PlayBoard:

    def __init__(self, l_x, l_y, d_x, d_y, striker, puck, simulation_ratio, cuda=False):
        """
        l_x: width of the playboard
        l_y: length of the playboard
        d_x: minimum x distance between striker and the edge
        d_y: maximum y distance between striker and the edge
        """
        self.L_x = l_x
        self.L_y = l_y
        self.D_x = d_x
        self.D_y = d_y
        self.Striker = striker
        self.Puck = puck
        self.decision = FCNet.FCNet().cuda()
        self.cuda = cuda
        self._simulation_ratio = simulation_ratio

    def new_game(self):
        self.decision.zero_grad()

        self.Striker.new_game()
        self.Puck.new_game()

    def move_step(self):
        state = [self.Puck.x, self.Puck.y, self.Puck.vx, self.Puck.vy,
                 self.Striker.x, self.Striker.y, self.Striker.vx, self.Striker.vy]

        if self.cuda:
            pass
        else:

            state = torch.tensor(state)  # .cuda()
            new_striker_v = self.decision(state)
            new_striker_v = torch.mul(new_striker_v, self.Striker.Max_velocity)

            self.Striker.new_decision(new_striker_v)

            self.Striker.x += self.Striker.vx * self._simulation_ratio
            self.Striker.y += self.Striker.vy * self._simulation_ratio

            self.Puck.x += self.Puck.vx * self._simulation_ratio
            self.Puck.y += self.Puck.vy * self._simulation_ratio

        if self.strike_check():
            return False
        else:
            return True

    def strike_check(self):
        pass

    def run_till_strike(self):

        while self.move_step():
            pass


if __name__ == "__main__":
    pass
