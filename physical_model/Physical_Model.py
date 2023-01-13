from DL_model import FCNet
import torch
from defensive_control_strategy.defensive_control_strategy import Parameters


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
        self.optimizer = torch.optim.Adam(self.decision.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    def new_game(self):
        self.decision.zero_grad()

        self.Striker.new_game()
        self.Puck.new_game()

    def simulation_step(self):

        if self.cuda:
            pass
        else:

            state = [self.Puck.x, self.Puck.y, self.Puck.vx, self.Puck.vy, self.Striker.x, self.Striker.y,
                     self.Striker.vx, self.Striker.vy]
            state = torch.tensor(state)  # .cuda()
            state = torch.reshape(state, (1, 8))
            y = self.decision(state)

            new_striker_v_mu = torch.mul(y, self.Striker.Max_velocity)
            new_striker_v = torch.normal(new_striker_v_mu, torch.tensor([Parameters.sigma, Parameters.sigma]))

            self.Striker.new_decision(new_striker_v)

            self.Striker.x += self.Striker.vx * self._simulation_ratio
            self.Striker.y += self.Striker.vy * self._simulation_ratio

            self.Puck.x += self.Puck.vx * self._simulation_ratio
            self.Puck.y += self.Puck.vy * self._simulation_ratio

            return state, new_striker_v

    def bp_step(self, state_stack, action_stack):  # TODO

        criterion = FCNet.MonteCarloPolicyGradientLossFunc()
        loss = criterion(self.output, self.end_punishment())

        loss.backward()

        self.optimizer.step()

    def end_check(self):

        if self.Puck.y <= self.Striker.y:
            return True
        else:
            distance = (self.Puck.x - self.Striker.x) ^ 2 + (self.Puck.y - self.Striker.y) ^ 2

            if distance <= (self.Puck._radius + self.Striker._radius):
                return True
            else:
                return False

    def end_punishment(self):  # TODO: Reward
        P = abs(self.Puck.x - self.Striker.x)

        if not self.Puck.y <= self.Striker.y:
            P *= 2
        return P

    def run_till_strike(self):

        state_stack = torch.empty(0, 8)
        action_stack = torch.empty(0, 2)
        while not self.end_check():
            state, new_striker_v = self.simulation_step()
            state_stack = torch.cat((state_stack, state), 0)
            action_stack = torch.cat((action_stack, new_striker_v), 0)

        self.bp_step(state_stack, action_stack)

        return self.end_punishment()  # TODO


if __name__ == "__main__":
    pass
