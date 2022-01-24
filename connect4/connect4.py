import typing
import math
import re
import numpy as np
import keras
import tensorflow as tf
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import igame

from connect4.c4_Model import create_model
from player import Player

row_count, col_count = 6, 7
maxMoves = row_count * col_count


class Connect4Board(igame.GameBoardBase):
    def __init__(self):
        self.board = np.zeros((row_count, col_count), dtype = int)
        self.move_count = 0
        self.next_player = Player.P1

    def copy_board_to(self, target_board):
        np.copyto(target_board.board, self.board)
        target_board.move_count = self.move_count
        target_board.next_player = self.next_player

    def get_current_State(self):
        
        p1_idx = 0 if self.next_player == Player.P1 else 1
        p2_idx = 1 - p1_idx

        tensor = np.zeros((row_count, col_count, 2), dtype = np.float32)
        tensor[:, :, p1_idx] = self.board * (self.board > 0)
        tensor[:, :, p2_idx] = -self.board * (self.board < 0)
        
        return tensor

    def get_possible_moves(self):
        colNotFull = np.sum(np.abs(self.board), axis = 0) < row_count
        return (i for i in range(col_count) if colNotFull[i])

    def parse_str_move(self, input_str):
        try:
            return int(input_str)
        except ValueError:
            return None

    def make_move(self, move):
        j = move
        i = row_count - 1
        while self.board[i, j] != 0:
            i -= 1

        next_playerValue = self.next_player.to_state_val()
        self.board[i, j] = next_playerValue

        playerWon = False
        for (di, dj) in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            consCount = 0
            for inc in range(-3, 4):
                i2, j2 = i + di * inc, j + dj * inc
                if i2 < 0 or i2 >= row_count:
                    continue
                if j2 < 0 or j2 >= col_count:
                    continue
                if self.board[i2, j2] != next_playerValue:
                    consCount = 0
                else:
                    consCount += 1
                    if consCount == 4:
                        playerWon = True
                        break
            if playerWon:
                break

        self.move_count += 1

        if playerWon:
            return True, self.next_player
        elif self.move_count == maxMoves:
            return True, None

        self.next_player = self.next_player.opposite()
        return False, None

    def get_next_player(self): 
        return self.next_player

    def pretty_board(self, policy = None, value = None):


        firstLine = "--- Next Player: "
        if self.next_player == Player.P1: firstLine += "O"
        else:                                 firstLine += "X"

        if value is not None:
            firstLine += ", Value = %f" % value

        lines = [firstLine]

        if policy is not None:
            lines.append(" ".join((("%3d" % (int(p * 999.99)) if p > 0 else "   ") for p in policy)))

        moves_strs = [" . ", " O ", " X "]
        for i in range(row_count):
            lines.append(" ".join(moves_strs[self.board[i, j]] for j in range(col_count)))

        return "\r\n".join(lines)
    

def augment_data(states, policies, values):
    
    states = np.concatenate([states, states[:, :, ::-1, :]])
    
    policies = np.concatenate([policies, policies[:, ::-1]])
    
    values = np.concatenate([values, values])
    
    return states, policies, values

TrainConfig = igame.TrainConfig(
    bundle_count = 3,
    games_per_bundle = 60,
    batch_size = 32,
    epochs_per_iteration = 256, 
    evalgames_per_iteration = 16,)

class Connect4(igame.GameBase):
    def get_game_name():
        return "connect4"
    def get_train_params():
        return TrainConfig
    def get_mcts_simulation_number():
        return 128  
    def get_policy_len():
        return col_count
    def to_policy_index(move):
        return move
    def to_move(p):
        return p
    def create_board():
        return Connect4Board()
    def create_new_model():
        return create_model()
    def augment_data(states, policies, values):
        return augment_data(states, policies, values)


