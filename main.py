import argparse
import pipeline
import play_vs_net

parser = argparse.ArgumentParser()
parser.add_argument("--action", default="train", type=str, nargs="?", help="[train/play] to train or play with model")
parser.add_argument("--game", default="connect4", type=str, nargs="?",help="[connect4/ttt] to pick specific game connect4 or tictactoe")
parser.add_argument("--player-mark", default="connect4", type=str, nargs="?",help="[x/X/o/O] to pick specific side of game")
args = parser.parse_args()   




if args.action == "train":
    pipeline.run_pipeline(args)
else:
    play_vs_net.run_pve(args)       