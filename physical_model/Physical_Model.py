class PlayBoard:

    def __init__(self, l_x, l_y, d_x, d_y, striker, puck):
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

    def new_game(self):
        pass

    def move_step(self):
        pass

    def strike_check(self):
        pass

    def run(self):
        pass


if __name__ == "__main__":
    pass
