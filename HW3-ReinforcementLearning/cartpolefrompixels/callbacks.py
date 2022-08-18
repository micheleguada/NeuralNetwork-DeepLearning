import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback



class MaxEpisodesStop(Callback):    
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        
        if module.episode_id >= module.N_episodes:
            print(f"Reached number of episodes ({module.N_episodes}). Ending...")
            trainer.should_stop = True
            
        return
    
    
class RLResults(Callback):
    """
    Agent results
    """
    def __init__(self, name="results"): 
        self.name  = name
        self.current_id = 0 
        
        self.ep_ids       = []
        self.losses       = []
        self.rewards      = []
        self.scores       = []
        self.temperatures = []
    
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):  
        
        try:
            results = trainer.logged_metrics[self.name]
            
            if self.current_id != results["episode_id"].item():
                self.ep_ids.append( results["episode_id"].item() )
                self.losses.append( results["final_loss"].item() )
                self.rewards.append( results["final_reward"].item() )
                self.temperatures.append( results["temperature"].item() )
                self.scores.append( results["score"].item() )
                
                self.current_id = results["episode_id"].item()
                
        except KeyError:
            #print("Epoch ended before at least one episode was completed.")
            pass
            
        return
    
