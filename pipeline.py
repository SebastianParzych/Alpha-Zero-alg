import sys
import os
import shutil
from evaluate import evaluate_net
import logging
import time
import collections
import numpy as np
import keras
import mtcs
import igame  
from logs.logger import logger
import matplotlib.pyplot as plt

currDir = os.path.dirname(os.path.realpath(__file__))
    
def load_models():
    # load current model
    if os.path.isfile(modelPath):
        logging.info("Loading existing model")
        loaded_model = keras.models.load_model(modelPath)
        model = game.create_new_model()
        model.set_weights(loaded_model.get_weights())
    else:
        logging.info("Creating new model")
        model = game.create_new_model()
    # load best model 
    if os.path.isfile(bestModelPath):
        logging.info("Loading existing best model")
        best_model = keras.models.load_model(bestModelPath)
    else:
        logging.info("Duplicating new best model")
        best_model = keras.models.clone_model(model)
        best_model.set_weights(model.get_weights())  
          
    return model , best_model


def generate_data(model, games_per_iteration):
    batch_states = []
    batch_policies = []
    batch_values = []
    for i in range(games_per_iteration):
        states, policies, values = mtcs.run_selfplay(model, game, 1)
        batch_states.extend(states)
        batch_policies.extend(policies)
        batch_values.extend(values)
        logger.info("MCTS self play generated game nr: %s/%s" % (i+1,games_per_iteration))
    return batch_states, batch_policies, batch_values


class TrainingCallback(keras.callbacks.Callback):
    
            def formatLogs(self, logs = None):
                if logs is not None: self.logs = logs
                outstr = ", ".join(["%s: %0.3f" % pair for pair in sorted(self.logs.items())])
                return outstr
            def set_2params(self, params):
                self.epochs = params["epochs"]
                return super().set_params(params)
            def on_epoch_end(self, epoch, logs = None):
                return super().on_epoch_end(epoch, logs)
            def on_train_end(self, logs = None):
                return super().on_train_end(logs)  
            
            
def training(args):
    global game
    game = args.game
    

    
    if game == 'ttt':
        from tictactoe.tictactoe import TicTacToe as game
    else:
        from connect4.connect4 import Connect4 as game
        
    assert issubclass(game, igame.GameBase)

    game_name = game.get_game_name()
    print(game_name)
    
    global selfPlayDataPath; global backupSelfPlayDataPath; global modelPath
    global backupModelPath; global bestModelPath; global backupBestModelPath
    global trainingDataPath
    
    selfPlayDataPath = os.path.join(currDir, "model_data/_" + game_name + ".selfPlay.npz")
    backupSelfPlayDataPath = os.path.join(currDir, "model_data/_" + game_name + ".selfPlay.backup.npz")
    modelPath = os.path.join(currDir, "model_data/_" + game_name + ".model.h5")
    backupModelPath = os.path.join(currDir, "model_data/_" + game_name + ".model.backup.h5")
    bestModelPath = os.path.join(currDir, "model_data/_" + game_name + ".bestModel.h5")
    backupBestModelPath = os.path.join(currDir, "model_data/_" + game_name + ".bestModel.backup.h5")
    trainingDataPath = os.path.join(currDir,"train_data")
    
    if os.path.exists(os.path.join(currDir,"model_data")):
        pass
    else:
        os.makedirs(os.path.join(currDir,"model_data"))
        
    if os.path.exists(trainingDataPath):
        pass
    else:
        os.makedirs(trainingDataPath)
    
    model, best_model = load_models()
    logger.info("Starting Training Pipeline...")
    params = game.get_train_params()   
    logger.info(params)
    epoch_number = 0
    iteration = 0
    data_buffer = collections.deque()

    if os.path.isfile(selfPlayDataPath):
        logger.info("Loading existing %d data from last iteration" % params.bundle_count)
        npz = np.load(selfPlayDataPath)
        for i in range(params.bundle_count):
            data_buffer.append((npz["s%d" % i], npz["p%d" % i], npz["v%d" % i]))
        if ("iteration" not in npz) and (game.get_game_name() == "tictactoe"):
            iteration = 0
        else:
            iteration = int(npz["iteration"])
    else:
        logger.info("Initializing %d data bundles...." % params.bundle_count)
        for i in range(params.bundle_count):
            logger.info("Creating bundle : %s/%s" % (i+1, params.bundle_count))
            data_buffer.append(generate_data(best_model, params.games_per_bundle // 2))
    logger.info("%s size of generated data" % (len(data_buffer)))        

    for i in range(30):
        logger.info("Starting current %i training iteration" % (i))
        logger.info("Training total iteration %d " % iteration)
        npz_e = {"iteration": iteration}
        for j in range(len(data_buffer)):
            states, policies, values = data_buffer[j]
            npz_e.update({("s%d" % j): states, ("p%d" % j): policies, ("v%d" % j): values})
        try:
            np.savez(selfPlayDataPath, **npz_e)
            shutil.copyfile(selfPlayDataPath, backupSelfPlayDataPath)
        except BaseException as err:
            logger.error(err)

        logger.info("Prepearing training dataset ... ")
        
        states   = np.array([state  for (states, _, _) in data_buffer for state  in states])
        policies = np.array([policy for (_, policies, _) in data_buffer for policy in policies])
        values   = np.array([value  for (_, _, values) in data_buffer for value  in values])
        
        logger.info("Prepearsed samples: (%d, %d, %d) states,policies,values" % (len(states), len(policies), len(values)))
        states, policies, values = game.augment_data(states, policies, values)
        logger.info("Augmented data: (%d, %d, %d)" % (len(states), len(policies), len(values)))
        logger.info("Train model, %d epochs" % params.epochs_per_iteration)

            
        model_history=model.fit(
            states, [policies, values],
            batch_size = params.batch_size,
            epochs = epoch_number + params.epochs_per_iteration,
            initial_epoch = epoch_number,
            callbacks = [TrainingCallback()],
            verbose = 1)
        
        #print(vars(model_history))
        np.save(trainingDataPath+'/%s_%s' % (game_name,i), model_history.history)
        #fig, ax = plt.subplots(3, 2)
        ax1 = plt.subplot2grid(shape=(3,3), loc=(0,0), colspan=3)
        ax2 = plt.subplot2grid((3,6), (1,0), colspan=3)
        ax3 = plt.subplot2grid((3,6), (1,3), colspan=3)
        ax4 = plt.subplot2grid((3,6), (2,0), colspan=3)
        ax5 = plt.subplot2grid((3,6), (2,3), colspan=3)

        ax1.set_title('Funkcja kosztu', size=9)
        ax2.set_title('Funkcja kosztu głowy polityki', size=9)
        ax3.set_title('Dokładność głowy polityki', size=9)
        ax4.set_title('Funkcja kosztu głowy wartości', size=9)
        ax5.set_title('Dokładność głowy wartości', size=9)


        ax1.set_xlabel('epoki', size=9)
        ax1.tick_params(axis="both", labelsize=7) 
        ax2.set_xlabel('epoki', size=9)
        ax2.tick_params(axis="both", labelsize=7) 
        ax3.set_xlabel('epoki', size=9)
        ax3.tick_params(axis="both", labelsize=7) 
        ax4.set_xlabel('epoki', size=9)
        ax4.tick_params(axis="both", labelsize=7) 
        ax5.set_xlabel('epoki', size=9)
        ax5.tick_params(axis="both", labelsize=7) 

        ax1.plot(model_history.epoch, model_history.history['loss'])
        ax2.plot(model_history.epoch, model_history.history['policy_head_loss'])
        ax3.plot(model_history.epoch, model_history.history['policy_head_accuracy'])
        ax4.plot(model_history.epoch, model_history.history['value_head_loss'])
        ax5.plot(model_history.epoch, model_history.history['value_head_accuracy'])
        
        plt.tight_layout(pad=2.0)

        plt.savefig(trainingDataPath+'/%s_%s' % (game_name,i))   

        epoch_number += params.epochs_per_iteration
        
        try:
            model.save(modelPath, overwrite=True)
            shutil.copyfile(modelPath, backupModelPath)
        except BaseException as err:
            logger.error(err)
        logger.info("Challenge current vs  best model")
        
        net_improved = evaluate_net(params, current_net=model,best_net=best_model,game=game)
        
        if net_improved:
            logger.info("Challenge succeeded")
            best_model.set_weights(model.get_weights())
        else:
            logger.info("Challenge failed")
            
        if net_improved or not os.path.isfile(bestModelPath):
            try:
                best_model.save(bestModelPath, overwrite=True)
                shutil.copyfile(bestModelPath, backupBestModelPath)
            except BaseException as err:
                logger.error(err)
        
        logger.info("Update new self play data bundle")
        data_buffer.pop()
        data_buffer.appendleft(generate_data(best_model, params.games_per_bundle))
        iteration += 1

def run_pipeline(args):

    training(args)
    

  



