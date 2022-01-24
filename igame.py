"""Interface for games"""
import collections

class GameBoardBase:
    def copy_board_to(self, target_board):
        """Copy from current board to new board all params"""
        raise NotImplementedError()
    def get_current_State(self): 
        raise NotImplementedError()
    def get_possible_moves(self):
        raise NotImplementedError()
    def parse_str_move(self, inputStr):
        raise NotImplementedError()
    def make_move(self, move): 
        raise NotImplementedError()
    def get_next_player(self):
        raise NotImplementedError()
    def pretty_board(self, policy = None, value = None):
        raise NotImplementedError()

TrainConfig = collections.namedtuple(
    "TrainConfig",["bundle_count",
                      "games_per_bundle",
                      "batch_size",
                      "epochs_per_iteration",
                      "evalgames_per_iteration"])

class GameBase:
    def get_game_name():
        raise NotImplementedError()
    def get_train_params():
        raise NotImplementedError()
    def get_mcts_simulation_number():
        raise NotImplementedError()
    def get_policy_len():
        raise NotImplementedError()
    def to_policy_index(move):
        raise NotImplementedError()
    def to_move(pi):
        raise NotImplementedError()
    def create_board():
        raise NotImplementedError()
    def create_new_model():
        raise NotImplementedError()
    def augment_data(states, policies, values):
        raise NotImplementedError()