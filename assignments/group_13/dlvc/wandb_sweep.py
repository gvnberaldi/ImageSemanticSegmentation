import torch
import wandb


class WandBHyperparameterTuning:
    def __init__(self,  api_key: str, project_name: str, entity_name: str = None):
        """
        Initialize the class with a W&B project name and (optionally) entity name.
        """
        self.project_name = project_name
        self.entity_name = entity_name
        self.sweep_id = None
        self.sweep_config = None
        self.training_function = None

        wandb.login(key=api_key)

    def set_sweep_config(self, metric_name, metric_goal, hyperparameters, method='random'):
        """
        Define the sweep configuration for hyperparameter tuning.
        """
        self.sweep_config = {
            'method': method,
            'metric': {
                'name': metric_name,
                'goal': metric_goal
            },
            'parameters': {
                key: {'values': value} if type(value) is list else value for key, value in hyperparameters.items()
            }
        }

    def create_sweep(self):
        """
        Create a sweep with the current configuration and return the sweep ID.
        """
        if not self.sweep_config:
            raise ValueError("Sweep configuration is not set. Call set_sweep_config first.")

        self.sweep_id = wandb.sweep(self.sweep_config, project=self.project_name, entity=self.entity_name)
        return self.sweep_id

    def set_training_function(self, training_function):
        """
        Set the training function to be used during the sweep.
        """
        self.training_function = training_function

    def run_sweep(self, count=10):
        """
        Run the sweep with the specified count of runs.
        """
        if not self.sweep_id:
            raise ValueError("Sweep ID is not set. Call create_sweep first.")

        if not self.training_function:
            raise ValueError("Training function is not set. Call set_training_function first.")

        wandb.agent(self.sweep_id, function=self.training_function, count=count)

    def log(self, log_dict: dict, commit=True, step=None):
        if step:
            wandb.log(log_dict, commit=commit, step=step)
        else:
            wandb.log(log_dict, commit=commit)

