class Parameters:

    def __init__(self):
        pass

    Simulation_ratio = 0.02  # Hz-1
    sigma = 0.5  # gauss distribution of strike velocity

    class Striker:
        Radius = 8.0
        Max_velocity = 25  # per s, one dimension

    class Puck:
        Radius = 2.0
        Max_velocity = 50  # per s, one dimension

    class PlayBoard:
        L_x = 50.0
        L_y = 100.0
        D_x = 10.0  # minimum distance between Striker and longitudinal edge
        D_y = 15.0  # minimum distance between Striker and Horizontal edge

    cuda = True


