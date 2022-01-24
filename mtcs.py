import math
import numpy as np
import keras
from igame import *
from player import Player


def upgrade_model_policy(policy, tau):
    policy = np.log(policy + 1e-10) / tau
    policy = np.exp(policy - np.max(policy))
    return policy / np.sum(policy)

def get_weighted_rand(policy):
    return np.random.choice(policy.size, p = policy)


class Node:
    def __init__(self, parent, p):
        self.parent = parent
        self.P = p
        self.N = 0
        self.W = 0
        self.Q = 0
        self.children = None

    def get_U(self, cpuct):
        N_sum = self.N if self.parent == None else self.parent.N
        if N_sum == 0:
            N_sum = 1
        return cpuct * self.P * math.sqrt(N_sum) / (1 + self.N)

    def get_UCB(self, cpuct):
        return self.Q + self.get_U(cpuct)

    def expand(self, board, game, childPolicies, is_explored):
        policies = np.array([*(childPolicies[game.to_policy_index(move)] for move in board.get_possible_moves())])
        policies = policies / np.sum(policies)
        if is_explored:
            # Dirichlet noise
            epsilon = 0.25
            noise = np.random.dirichlet((0.03, 0.97), size=policies.shape)[:, 0]
            policies = policies * (1.0 - epsilon) + noise * epsilon
            #normalize
            policies = policies / np.sum(policies)

        self.children = dict(zip(board.get_possible_moves(), (Node(self, policy) for policy in policies)))

    def backpropagate(self, node_val):
        self.N += 1
        self.W += node_val
        self.Q = self.W / self.N


def mtcs(root, prev_player, board, iter_board, model, game, is_explored):
   
    if root.children == None:
        policy, value = model.predict(np.array([board.get_current_State()]))
        root.expand(board, game, policy[0, :], is_explored)

    root_player = board.get_next_player()
    root_val = root.W if root_player == prev_player else - root.W
    cpuct = np.sqrt(game.get_mcts_simulation_number()) / 8
    
    for _ in range(game.get_mcts_simulation_number()):

        board.copy_board_to(iter_board)
        iter_node = root
        curr_players = []
        while bool(iter_node.children):
            # Get move due to ucb value
            move = max(iter_board.get_possible_moves(),
                key = lambda move: iter_node.children[move].get_UCB(cpuct))
            
            iter_node = iter_node.children[move]
            currPlayer = iter_board.get_next_player()
            curr_players.append(currPlayer)
            is_game_ended, winner = iter_board.make_move(move)

        node_val = 0
        if is_game_ended:
            if winner == None:
                node_val = 0
            elif winner == currPlayer:
                node_val = 1
            else: 
                node_val = -1
        else:
            policy, value = model.predict(np.array([iter_board.get_current_State()]))
            iter_node.expand(iter_board, game, policy[0, :], is_explored)
            node_val = -value[0, 0]
            
        
        for player in reversed(curr_players):
            iter_node_val = node_val if player == currPlayer else -node_val
            iter_node.backpropagate(iter_node_val)
            iter_node = iter_node.parent
        
        root.N += 1

        root_val += node_val if root_player == currPlayer else -node_val


    policy = np.zeros((game.get_policy_len()), dtype = np.float32)
    for move in board.get_possible_moves():
        policy[game.to_policy_index(move)] = root.children[move].N
    policy = policy / np.sum(policy)
    value = root_val / root.N
    return policy, value

def run_selfplay(model, game, temperature = 0.001):

    states = []
    policies = []
    values = []
    players = []

    root = Node(None, 1)
    board = game.create_board()
    iter_board = game.create_board()
    

    
    prev_player = Player.P1
    is_gamee_ended, winner = False, None
    while not is_gamee_ended:
        
        policy, value = mtcs(root, prev_player, board, iter_board, model, game, True)
        states.append(board.get_current_State())
        policies.append(policy)
        values.append(value)
        players.append(board.get_next_player())

        prev_player = board.get_next_player()
        policy = upgrade_model_policy(policy, temperature)
        move = game.to_move(get_weighted_rand(policy))
        is_gamee_ended, winner = board.make_move(move)
        root = root.children[move]
        root.parent = None
        
    final_vals = []
    for player in players:
        if winner is None:
            final_vals.append(0)
        elif winner == player: 
            final_vals.append(1)
        else:
            final_vals.append(-1)

    return states, policies, final_vals

