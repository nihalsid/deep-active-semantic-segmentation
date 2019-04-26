import os
import shutil
import torch
from collections import OrderedDict
import glob
import constants
import json


class Saver:

    def __init__(self, args, experiment_group=None, remove_existing=False):

        self.args = args

        if experiment_group == None:
            experiment_group = args.dataset

        self.directory = os.path.join(constants.RUNS, experiment_group, args.checkname)
        self.experiment_dir = self.directory

        if remove_existing and os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)

        if not os.path.exists(self.experiment_dir):
            print(f'making dir {self.experiment_dir}')
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):

        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def save_experiment_config(self):

        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        arg_dictionary = vars(self.args)
        log_file.write(json.dumps(arg_dictionary, indent=4, sort_keys=True))
        log_file.close()


class ActiveSaver(Saver):

    def __init__(self, args, num_of_labeled_samples, experiment_group=None):

        super().__init__(args, experiment_group=experiment_group)
        self.experiment_dir = os.path.join(self.directory, f'run_{num_of_labeled_samples:04d}')

        if not os.path.exists(self.experiment_dir):
            print(f'making dir {self.experiment_dir}')
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):

        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
                f.write(f'\n{str(state["epoch"])}')

            filename = os.path.join(self.experiment_dir, 'best.pth.tar')
            torch.save(state, filename)

    def save_active_selections(self, paths):

        filename = os.path.join(self.experiment_dir, 'selections.txt')
        with open(filename, 'w') as fptr:
            for p in paths:
                fptr.write(p.decode('utf-8') + '\n')


class PassiveSaver(Saver):

    def __init__(self, args):

        super().__init__(args)

        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, f'experiment_{run_id}')

        if not os.path.exists(self.experiment_dir):
            print(f'making dir {self.experiment_dir}')
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):

        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
                f.write(f'\n{str(state["epoch"])}')

            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, f'experiment_{run_id}', 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
