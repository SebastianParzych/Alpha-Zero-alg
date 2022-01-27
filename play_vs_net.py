
from multiprocessing.sharedctypes import Value
import numpy as np
import keras
from igame import *
from player import Player
import os

from mtcs import (
    upgrade_model_policy,
    get_weighted_rand,
    Node,
    mtcs
)


currDir = os.path.dirname(os.path.realpath(__file__))


def play(p1_model, p2_model, game, verbose = False, tempurature = 0.05):


    board = game.create_board()
    iter_board = game.create_board()


    player_lookup = {
       Player.P1: (p1_model, p1_model and Node(None, 1)),
       Player.P2: (p2_model, p2_model and Node(None, 1))
       }

    prevPlayer = Player.P1
    isGameEnded, winner = False, None
    while not isGameEnded:
        player = board.get_next_player()
        assert isinstance(player, Player)

        model, root = player_lookup[player]
        move = None
        if model is None:
            print(board.pretty_board())
            while move is None:
                inputLine = input("--- Take your move: ")
                parsedMove = board.parse_str_move(inputLine)
                if parsedMove is None:
                    continue
                if parsedMove not in board.get_possible_moves():
                    continue
                move = parsedMove

        else:
            policy, value = mtcs(root, prevPlayer, board, iter_board, model, game, False)
            if verbose:
                print(board.pretty_board(policy, value))

            policy = upgrade_model_policy(policy, tempurature)
            move = game.to_move(get_weighted_rand(policy))

            root = root.children[move]
            root.parent = None
            player_lookup[player] = model, root

        player = player.opposite()
        model, root = player_lookup[player]
        if model is not None:
            if root.children == None:
                policy, _ = model.predict(np.array([board.get_current_State()]))
                root.expand(board, game, policy[0, :], False)

            root = root.children[move]
            root.parent = None
            player_lookup[player] = model, root

        prevPlayer = board.get_next_player()
        isGameEnded, winner = board.make_move(move)

    p1Won = None if winner is None else (winner == Player.P1)
    if verbose:
        print(board.pretty_board())
        print("--- Winner is %s \n" % winner)


    return p1Won


def is_Human(mark):
    print(mark)
    if mark.lower() == "x":
        return True
    elif mark.lower() =='o':
        return False
    else:
        raise ValueError("Incorrect player!")

def run_pve(args):
    
    if args.game == "ttt":
        
        from tictactoe.tictactoe import TicTacToe as game
    else:
        
        from connect4.connect4 import Connect4 as game
        
        
    game_name = game.get_game_name()
    model = None
    
    modelPath = os.path.join(currDir, "model_data/_" + game_name + ".model.h5")
    bestModelPath = os.path.join(currDir, "model_data/_" + game_name + ".bestModel.h5")
    if os.path.isfile(modelPath):
        print("Loading existing best model")
        model = keras.models.load_model(modelPath)
    else:
        print("Creating new model")
        model = game.create_new_model()


    p1_humnan=is_Human(args.player_mark)
    p1 = None if p1_humnan else model
    if not p1_humnan:
        p2_hmuman= True
    else:
        p2_hmuman= False
    p2 = None if p2_hmuman else model

    playAgain = True
    while playAgain:
        print("****************************************")
        print("Game start")
        play(p1, p2, game, True)

        print("****************************************")
        playAgain = None
        while playAgain is None:
            inputLine = input(">>> Play again? ([y]/n) ").strip().lower()
            if inputLine in ("", "y", "yes"):
                playAgain = True
            elif inputLine in ("n", "no"):
                playAgain = False
                