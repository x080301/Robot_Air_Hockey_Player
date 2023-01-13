from DL_model import FCNet
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
        self.optimizer = torch.optim.Adam(self.decision.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

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
            self.output = self.decision(state)

            new_striker_v_mu = torch.mul(self.output, self.Striker.Max_velocity)
            # new_striker_v = torch.normal(new_striker_v_mu, torch.tensor([Parameters.sigma, Parameters.sigma]))
            # TODO: It should be gauss distribution
            new_striker_v = new_striker_v_mu

            self.Striker.new_decision(new_striker_v)

            self.Striker.x += self.Striker.vx * self._simulation_ratio
            self.Striker.y += self.Striker.vy * self._simulation_ratio

            self.Puck.x += self.Puck.vx * self._simulation_ratio
            self.Puck.y += self.Puck.vy * self._simulation_ratio

    def end_check(self):

        if self.Puck.y <= self.Striker.y:
            return True
        else:
            distance = (self.Puck.x - self.Striker.x) ^ 2 + (self.Puck.y - self.Striker.y) ^ 2

            if distance <= (self.Puck._radius + self.Striker._radius):
                return True
            else:
                return False

    def end_punishment(self):
        P = abs(self.Puck.x - self.Striker.x)

        if not self.Puck.y <= self.Striker.y:
            P *= 2
        return P

    def update_decision_model(self):

        criterion = FCNet.MonteCarloPolicyGradientLossFunc()
        loss = criterion(self.output, self.end_punishment())

        loss.backward()

        self.optimizer.step()

    def run_till_strike(self):

        while not self.end_check():
            self.move_step()

        return self.end_punishment()


if __name__ == "__main__":
    pass
