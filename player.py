import enum


class Player(enum.Enum):
    P1 = 1
    P2 = 2

    def to_state_val(self):
        return 1 if self == Player.P1 else -1

    def opposite(self):
        return Player.P2 if self == Player.P1 else Player.P1