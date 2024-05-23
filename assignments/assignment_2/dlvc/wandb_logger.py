import torch
import wandb

#Make sure to add a valid key
wandb.login(key="place_your_key_here")
class WandBLogger:

    def __init__(self, enabled=True, 
                 model: torch.nn.modules=None, 
                 run_name: str=None) -> None:
        
        self.enabled = enabled



        if self.enabled:
            #Make sure to add a valid entity name
            wandb.init(entity="place_your_entity_here", 
                        project="assignment_2",
                        group="initial_testing")
            if run_name is None:
                wandb.run.name = wandb.run.id    
            else:
                wandb.run.name = run_name  

            if model is not None:
                self.watch(model)         
            
    def watch(self, model, log_freq: int=1):
        wandb.watch(model, log="all", log_freq=log_freq)
            

    def log(self, log_dict: dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)
 

    def finish(self):
        if self.enabled:
            wandb.finish()
