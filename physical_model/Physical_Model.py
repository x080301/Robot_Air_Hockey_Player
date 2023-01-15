from DL_model import FCNet
import torch
# from defensive_control_strategy.defensive_control_strategy import Parameters
from physical_model.Parameters import Parameters
import math


class PlayBoard:
    HIT_WALL = 0
    MISS_PUCK = 1
    STRIKE = 2
    ON_GONGING = 3

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

        self._simulation_ratio = simulation_ratio

        self.cuda = cuda

        if self.cuda:
            self.decision = FCNet.FCNet().cuda()
        else:
            self.decision = FCNet.FCNet()

        self.optimizer = torch.optim.Adam(self.decision.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    def new_game(self):

        self.Striker.new_game()
        self.Puck.new_game()

    def restore_checkpoint(self, checkpointfile):
        ckp = torch.load(checkpointfile, 'cuda')
        self.decision.load_state_dict(ckp['state_dict'])

    def simulation_step(self):

        state = [self.Puck.x, self.Puck.y, self.Puck.vx, self.Puck.vy, self.Striker.x, self.Striker.y,
                 self.Striker.vx, self.Striker.vy]

        state = torch.tensor(state)  # .cuda()
        if self.cuda:
            state = state.cuda()
        state = torch.reshape(state, (1, 8))

        # y = self.decision(state)
        # new_striker_v_mu = torch.mul(y, self.Striker.Max_velocity)
        new_striker_v_mu = self.decision(state)

        if self.cuda:
            new_striker_v = torch.normal(new_striker_v_mu, torch.tensor([Parameters.sigma, Parameters.sigma]).cuda())
        else:
            new_striker_v = torch.normal(new_striker_v_mu, torch.tensor([Parameters.sigma, Parameters.sigma]))

        self.Striker.new_decision(new_striker_v)

        self.Striker.x += self.Striker.vx * self._simulation_ratio
        self.Striker.y += self.Striker.vy * self._simulation_ratio

        self.Puck.x += self.Puck.vx * self._simulation_ratio
        self.Puck.y += self.Puck.vy * self._simulation_ratio

        return state, new_striker_v

    def bp_step(self, state_stack, action_stack):

        distance = math.sqrt((self.Puck.x - self.Striker.x) ** 2 + (self.Puck.y - self.Striker.y) ** 2)
        interval = distance - self.Puck._radius - self.Striker._radius

        if interval > 0:
            reward = 10.0 / (interval ** 2 + abs(self.Puck.x - self.Striker.x) + 1)
        else:
            reward = 10.0 / (abs(self.Puck.x - self.Striker.x) + 1)
            # 10.0 + self.Puck._radius + self.Striker._radius - abs(self.Puck.x - self.Striker.x)

        y = self.decision(state_stack)
        loss = self.decision.loss_function(y, action_stack, reward * 20)

        loss.backward()

        # self.optimizer.step()

    def end_check(self):

        if self.Puck.y <= self.D_y:
            return self.HIT_WALL
        elif self.Puck.y <= self.Striker.y:
            return self.MISS_PUCK
        else:
            distance = math.sqrt((self.Puck.x - self.Striker.x) ** 2 + (self.Puck.y - self.Striker.y) ** 2)

            if distance <= (self.Puck._radius + self.Striker._radius):
                return self.STRIKE
            else:
                return self.ON_GONGING

    def run_test(self):

        state_stack = torch.empty(0, 8)
        action_stack = torch.empty(0, 2)
        if self.cuda:
            state_stack = state_stack.cuda()
            action_stack = action_stack.cuda()

        end_check_flag = self.end_check()
        while end_check_flag is self.ON_GONGING:
            state, new_striker_v = self.simulation_step()
            state_stack = torch.cat((state_stack, state), 0)
            action_stack = torch.cat((action_stack, new_striker_v), 0)

            end_check_flag = self.end_check()

        return abs(self.Puck.x - self.Striker.x), state_stack

    def run_till_gameover(self):

        state_stack = torch.empty(0, 8)
        action_stack = torch.empty(0, 2)
        if self.cuda:
            state_stack = state_stack.cuda()
            action_stack = action_stack.cuda()

        end_check_flag = self.end_check()
        while end_check_flag is self.ON_GONGING:
            state, new_striker_v = self.simulation_step()
            state_stack = torch.cat((state_stack, state), 0)
            action_stack = torch.cat((action_stack, new_striker_v), 0)

            end_check_flag = self.end_check()

        self.bp_step(state_stack, action_stack)

        return abs(self.Puck.x - self.Striker.x)

    def save_checkpoint(self, iteration, distance):

        torch.save({'state_dict': self.decision.state_dict()},
                   'checkpoints/checkpoint_{:04d}_{:.3f}.ckp'.format(iteration, distance))


if __name__ == "__main__":
    pass
