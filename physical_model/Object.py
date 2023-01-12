class Cylinder:
    vx = 0
    vy = 0
    x = 0
    y = 0

    def __init__(self, radius):
        self._radius = radius

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def set_velocity(self, vx, vy):
        self.vx = vx
        self.vy = vy


class Striker(Cylinder):
    pass


class Puck(Cylinder):
    pass


if __name__ == "__main__":
    pass
