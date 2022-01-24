import typing
import math
import re
import numpy as np
import keras
import tensorflow as tf
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import igame
import mtcs
from tictactoe.ttt_Model import create_model
from player import Player

main_dimension = 3
move_pool_count = main_dimension * main_dimension

def move_to_policy_index(i, j):
    return i * main_dimension + j
def to_move(pi):
    return pi // main_dimension, pi % main_dimension

class TicTacToeBoard(igame.GameBoardBase):
    
    def __init__(self):
        self.board = np.zeros((main_dimension, main_dimension), dtype = int)
        self.move_count = 0
        self.next_player = Player.P1

    def copy_board_to(self, target_board):
        np.copyto(target_board.board, self.board)
        target_board.move_count = self.move_count
        target_board.next_player = self.next_player

    def get_current_State(self):
        p1_idx = 0 if self.next_player == Player.P1 else 1
        p2_idx = 1 - p1_idx
        tensor = np.zeros((main_dimension, main_dimension, 2), dtype = np.float32)  # 3 x 3 x 2
        tensor[:, :, p1_idx] = self.board * (self.board > 0)
        tensor[:, :, p2_idx] = -self.board * (self.board < 0)
        return tensor

    def get_possible_moves(self):
        result = zip(*np.where(self.board == 0))
        return list(result)

    def parse_str_move(self, inputStr):
        return re.match("""^\s*\d+\s*,\s*\d+\s*$""", inputStr) and eval(inputStr)

    def make_move(self, move):
        
        i, j = move
        next_player_val = self.next_player.to_state_val()
        self.board[i, j] = next_player_val
        player_won = all([self.board[i, j] == next_player_val for i in range(main_dimension)])
        player_won |= all([self.board[i, j] == next_player_val for j in range(main_dimension)])
        player_won |= ((i == j) and
            all([self.board[i, i] == next_player_val for i in range(main_dimension)]))
        player_won |= ((i + j == main_dimension - 1) and
            all([self.board[i, main_dimension - 1 - i] == next_player_val for i in range(main_dimension)]))
        self.move_count += 1

        if player_won:
            return True, self.next_player
        
        elif self.move_count == move_pool_count:
            return True, None

        self.next_player = self.next_player.opposite()
        return False, None

    def get_next_player(self):
        return self.next_player

    def pretty_board(self, policy = None, value = None):


        firstLine = "--- Next Player: "
        if self.next_player == Player.P1: 
            firstLine += "X"
        else:                         
            firstLine += "O"

        if value is not None:
            firstLine += ", Value = %f" % value

        lines = [firstLine]

        moves_strs = ["  .   ", "  X   ", "  O   "]
        for i in range(main_dimension):
            line = ""
            for j in range(main_dimension):
                move = self.board[i, j]
                if (move == 0) and (policy is not None):
                    line += "(%3d) " % (round(policy[move_to_policy_index(i, j)] * 999.4))
                else:
                    line += moves_strs[move]
            lines.append(line)

        return "\r\n".join(lines)

def augment_data(states, policies, values):
    print(policies.shape, states.shape, values.shape)
    states = np.concatenate([
                            states,
                            states[:, ::-1, ::-1, :],
                            states[:, ::-1, :, :],
                            states[:, :, ::-1, :]]
                            )
    states = np.concatenate([
        states, np.rot90(states, axes = (1, 2))])

    policies = np.reshape(policies, (len(policies), main_dimension, main_dimension))
    policies = np.concatenate([
                            policies,             
                            policies[:, ::-1, ::-1],
                            policies[:, ::-1, :],
                            policies[:, :, ::-1]]
                            )
    
    policies = np.concatenate([
        policies, np.rot90(policies, axes = (1, 2))])
    
    policies = np.reshape(policies,
                          (len(policies),
                           move_pool_count))

    values = np.concatenate([values, values, values, values])
    values = np.concatenate([values, values])

    return states, policies, values

TrainConfig = igame.TrainConfig(
    bundle_count = 2,    
    games_per_bundle = 8,
    batch_size = 16,
    epochs_per_iteration = 12,
    evalgames_per_iteration = 10)

class TicTacToe(igame.GameBase):
    def get_game_name():
        return "tictactoe"
    def get_train_params():
        return TrainConfig
    def get_mcts_simulation_number():
        return 128  
    def get_policy_len():
        return move_pool_count
    def to_policy_index(move):
        return move_to_policy_index(*move)
    def to_move(p):
        return to_move(p)
    def create_board():
        return TicTacToeBoard()
    def create_new_model():
        return create_model()
    def augment_data(states, policies, values):
        return augment_data(states, policies, values)


    