import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import mtcs
from logs.logger import logger
import play_vs_net as pvp
import warnings
warnings.filterwarnings("ignore")

def evaluate_net(params, current_net, best_net, game):
    win_count = 0
    lose_count = 0
    draw_count = 0
    
    for i in range(params.evalgames_per_iteration // 2):
        result = pvp.play(current_net, best_net, game)
        if result is None:
            draw_count += 1
        elif result:
            win_count  += 1
        else:              
            lose_count += 1
        logger.info("Evaluate games: wins: %s, loses: %s, draws: %s"% ( win_count, lose_count, draw_count))
    for i in range(params.evalgames_per_iteration // 2, params.evalgames_per_iteration):
        result = pvp.play(best_net, current_net, game)
        if result is None:
            draw_count += 1
        elif not result:
            win_count  += 1
        else:
            lose_count += 1
        logger.info("Evaluate games: wins: %s, loses: %s, draws: %s"% ( win_count, lose_count, draw_count))
        
        
    is_improved = False
    
    if (win_count+lose_count) > 0 and (win_count/(win_count+lose_count)) >= 0.55:
        is_improved = True

    return is_improved
    
    
def evaluate_random_net(params, current_net, game):
    """ Evaluate net with ranom moves making player.
     
    Args:
        params ([tuple]): [description]
        current_net ([keras.models.model]): [description]
        game ([GameBoardBase]): [description]
    """    
    random_net = 'random'
    win_count = 0
    lose_count = 0
    draw_count = 0
    
    for i in range(params.evalgames_per_iteration // 2):
        result = mtcs.play(current_net,random_net, game)
        if result is None:
            draw_count += 1
        elif result:
            win_count  += 1
        else:              
            lose_count += 1
        logger.info("Evaluate games with random player: wins: %s, loses: %s, draws: %s"
                    % ( win_count, lose_count, draw_count))

    for i in range(params.evalgames_per_iteration // 2, params.evalgames_per_iteration):
        result = mtcs.play(random_net, current_net, game)
        if result is None:
            draw_count += 1
        elif not result:
            win_count  += 1
        else:
            lose_count += 1
        
        logger.info("Evaluate games with random player: %s, loses: %s, draws: %s" 
                    % ( win_count, lose_count, draw_count))
        
