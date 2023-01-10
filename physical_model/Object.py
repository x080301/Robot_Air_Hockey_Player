class Striker:
    vx = 0
    vy = 0

    def __init__(self, radius, x, y):
        self.radius = radius
        self.x = x
        self.y = y


class Puck:

    def __init__(self, radius, x, y, vx, vy):
        self.radius = radius
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy


if __name__ == "__main__":
    print("test")
