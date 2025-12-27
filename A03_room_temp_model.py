
class Room:
    def __init__(self, R, C_air, M_air, start_temp,  room_id):
        self.R = R
        self.C_air = C_air
        self.M_air = M_air
        self.T = start_temp
        self.time = 0
        self.room_id = room_id

    def temp_update(self, Q_Ex, Q_AC):
        delta_T = (Q_Ex + Q_AC) / (self.M_air * self.C_air)
        self.T += delta_T
        self.time += 1

