import json
import os
import subprocess
import torch

with open('basic_config.json', 'r') as f:
    base_config = json.load(f)
from allennlp.nn import util

def execute(command):
    p = subprocess.Popen(command.split())
    p.wait()


# We delete the previous copies and logs
execute('rm -rf ' + os.getcwd() + '/copies/*')
execute('rm -rf ' + os.getcwd() + '/logs/*')
# Here is the list of all classes
classes = ['GPE', 'DATE', 'ORG', 'EVENT', 'LOC', 'FAC', 'CARDINAL', 'QUANTITY', 'NORP', 'ORDINAL', 'WORK_OF_ART',
           'PERSON', 'LANGUAGE', 'LAW', 'MONEY', 'PERCENT', 'PRODUCT', 'TIME']

GPU = [0, 1, 2] * 6  # I did it this way but you can do using some different way
configs = list(zip(classes, GPU))
for random_seed in range(1, 5):
    base_config['dataset_reader']['random_seed'] = random_seed
    processes = []
    for subseries in range(len(classes) / 3):
        subconfigs = configs[(subseries * 3):(subseries * 3 + 3)]
        for config in subconfigs:
            # Here we edit the config for a particular experiment
            base_config['dataset_reader']['valid_class'] = config[0]
            base_config['dataset_reader']['drop_empty'] = False
            base_config['trainer']['cuda_device'] = config[1]
            this_dir = os.getcwd().split('/')[-1]

            copy_directory = os.getcwd()[:-(len(this_dir) + 1)] + '/copies/' + this_dir + '/pnet_' + config[
                0] + '_' + str(random_seed)

            if not os.path.exists(copy_directory):
                os.makedirs(copy_directory)

            # We create a copy of all the code and run it in this directorty.
            execute('rm -rf ' + copy_directory)
            execute('cp -r ' + os.getcwd() + '/base ' + copy_directory)

            model_directory = os.getcwd()[:-(len(this_dir) + 1)] + '/models/' + this_dir + '/pnet_' + config[
                0] + '_' + str(
                random_seed)

            if not os.path.exists(model_directory):
                os.makedirs(model_directory)

            # Here we save our config
            with open(copy_directory + '/config.json', 'w') as outfile:
                json.dump(base_config, outfile)

            # Here we delete old models if there are some
            cmd = 'rm -rf ' + model_directory
            p = subprocess.Popen(cmd.split())
            p.wait()

            # Then we run the model
            cmd = 'python3 ' + copy_directory + '/my_run.py train '
            cmd += copy_directory + '/config.json -s '
            cmd += model_directory
            p = subprocess.Popen(cmd.split())
            processes.append(p)

        for i, process in enumerate(processes):
            process.wait()
