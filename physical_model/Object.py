# import defensive_control_strategy.defensive_control_strategy as dcs
import random
from ..defensive_control_strategy.defensive_control_strategy import Parameters


class Cylinder:
    vx = 0
    vy = 0
    x = 0
    y = 0
    Max_velocity = 0

    def __init__(self, radius, Max_velocity):
        self._radius = radius
        self.Max_velocity = Max_velocity

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def set_velocity(self, vx, vy):
        self.vx = vx
        self.vy = vy


class Striker(Cylinder):

    def new_game(self):
        self.vx = 0
        self.vy = 0
        self.x = Parameters.PlayBoard.D_x / 2.0
        self.y = Parameters.PlayBoard.D_y / 4.0

    def new_decision(self, vxy):
        self.vx = vxy[0]
        self.vy = vxy[1]


class Puck(Cylinder):
    def new_game(self):
        self.x = random.random() * Parameters.PlayBoard.L_x
        self.y = (random.random() * 0.5 + 0.5) * Parameters.PlayBoard.L_y

        goal_point_x = (random.random() / 3.0 + 0.333) * Parameters.PlayBoard.L_x

        self.vy = -random.random() * self.Max_velocity
        self.vx = (self.x - goal_point_x) / self.y * self.vy


if __name__ == "__main__":
    pass
