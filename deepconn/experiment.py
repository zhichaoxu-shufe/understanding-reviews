import os
import sys 
import json
from abc import ABC, abstractmethod
from datetime import datetime 

import torch

# self.args.dataset
# self.args.log_dir
# self.args.use_pretrain
# self.args.pretrained_path
# self.args.log
# self.args.model_name
# self.args.verbose
# self.args.log_idx
# self.args.stats
# self.args.stats_idx
# self.args.epoch



# self.args.lr
# self.lr_decay
# self.decay_patience
# args.max_grad_norm
class Args(object):
    pass

def parse_args(config):
    args = Args()
    with open(config, 'r') as f:
        config = json.load(f)
    for name, val in config.items():
        setattr(args, name, val)

    return args

class Experiment(ABC):
    def __init__(self, args, dataloaders):
        self.uid = datetime.now().strftime("%m-%d_%H:%M:%S")
        self.updates = 0
        self.args = args

        # model
        self.model_name = None 

        # dataloader
        self.train_dataloader = dataloaders["train"] if dataloaders["train"] is not None else None
        self.valid_dataloader = dataloaders["valid"] if dataloaders["valid"] is not None else None
        self.test_dataloader = dataloaders["test"] if dataloaders["test"] is not None else None

        # output
        self.out_dir = None
        self.best_model_path = None 
        self.log_path = None

    def setup(self):
        """
        Make directory for log files and saving models
        """
        self._make_dir()

    def _make_dir(self):
        """
        Make out directory for log file and saving models
        """
        hyper_name = self.uid + "_" + self.args['model_name']
        
        out_dir = "./{}/{}/{}".format(
            self.args['log_dir'],
            self.args['dataset'],
            self.args['model_name'],
            self.uid
        )
        try:
            os.makedirs(out_dir)
        except OSError as exc:  # Python >2.5
                pass

        self.best_model_path = os.path.join(out_dir, "best_model.ckpt")
        self.log_path = os.path.join(out_dir, "log.txt")
        self.out_dir = out_dir

    def print_write_to_log(self, text):
        """
        print to the terminal & write to the log file
        """ 
        if self.args['log']:
            try:
                with open(self.log_path, "a") as f:
                    f.write(text + "\n")
            except IOError as e:
                print("Cannot write a line into {}".format(self.log_path))

        print(text)

    def build_writers(self):
        pass 

    def print_model_stats(self):
        if self.model is not None:
            self.print_write_to_log("List of all Trainable Variables") 
            for i, (name, params) in enumerate(self.model.named_parameters()):
                if params.requires_grad:
                    self.print_write_to_log("param {:3}: {:15} {}".format(i, str(tuple(params.shape)), name))
                else:
                    print("[Warning]: the parameters {} is not trainable".format(name))

            param_num = self._num_parameters()
            self.print_write_to_log("The total number of trainable parameters: {:,d}".format(param_num))
            self.print_write_to_log("="*50)
        else:
            raise ValueError("not found model")
    
    def print_args(self):
        for name, val in self.args.items():
            self.print_write_to_log("{}: {}".format(name, val))
        self.print_write_to_log("="*50)

    def _num_parameters(self):
        if self.model is not None:
            return sum([p.numel() for p in self.model.parameters()])
        else:
            raise ValueError("not found model")
    
    def save(self, name=None):
        if name is not None:
            if not name.endswith(".pt"):
                name += ".pt"
            fn = os.path.join(self.out_dir, name)
        else:
            fn = os.path.join(self.out_dir, "{}_model.pt".format(self.updates))
        
        params = {"model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "updates": self.updates,
                    "args": self.args}
        torch.save(params, fn)

    def load(self, fn):
        pass

    def update_stats(self, stats, set_name):
        """
        stats: Dict, 
        """
        if set_name == "train":
            for key, val in stats.items():
                self.train_stats[key] += val 
        elif set_name == "valid":
            for key, val in stats.items():
                self.valid_stats[key] += val
        else:
            raise ValueError(f"{set_name} is not predefined")
        
    def write_stats(self, set_name):
        fn = os.path.join(self.out_dir, "stats_{}.log.gz".format(set_name))
        if set_name == "train":
            with gzip.open(fn, "wt") as fzip:
                json.dump(self.train_stats, fzip) # see https://stackoverflow.com/questions/39450065/python-3-read-write-compressed-json-objects-from-to-gzip-file
        elif set_name == "valid":
            with gzip.open(fn, "wt") as fzip:
                json.dump(self.valid_stats, fzip)
        else:
            raise ValueError(f"{set_name} is not predefined")    
    
if __name__ == "__main__":
    config_file = "jjj.json"
    args = parse_args(config_file)
    
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Conv1d(100, 100, 3),
                            nn.ReLU(),
                            nn.Dropout(),
                            nn.Linear(100, 30),
                            nn.ReLU(),
                            nn.Dropout(),
                            nn.Linear(30, 3))
    datalaoders = {"train": None, "valid": None, "test": None}

    exp = Experiment(args, model, datalaoders)
    exp.setup()
    exp.print_args()
    exp.print_model_stats()