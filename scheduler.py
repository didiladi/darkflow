import numpy as np
import pandas as pd
from models import Model
import subprocess
import os
import time
import json
import os.path
import psutil

PROCNAME = "python3 ./flow"


def is_darkflow_running():

    for proc in psutil.process_iter():
        # check whether the process name matches
        if PROCNAME in proc.name():
            return True

    return False


def kill_darkflow():

    for proc in psutil.process_iter():
        # check whether the process name matches
        if PROCNAME in proc.name():
            proc.kill()
            break


def start_darkflow(model):

    process = '' + PROCNAME + \
        ' --model ' + model.get_config() + \
        ' --train' + \
        ' --dataset ' + model.get_path_for_train_images() + \
        ' --annotation ' + model.get_path_for_train_annotations()

    if model.get_ckpt_start() == 0:
        process = process + ' --load ' + model.get_start_weights()
    else:
        process = process + ' --load -1'

    os.spawnl(os.P_NOWAIT, process)


def is_model_finished(data, current_ckpt):

    if str(current_ckpt) not in data:
        return False

    for value in data[str(current_ckpt)]:
        if value < 0.7 or value > 0.99:
            return False

    return True


if __name__ == '__main__':

    darkflow_running = is_darkflow_running()

    for i in range(0, 9):

        model = Model(i+1)
        data = model.get_evaluation_data()
        current_ckpt = model.get_ckpt_start()

        if is_model_finished(data, current_ckpt):
            if darkflow_running:
                kill_darkflow()
            if i < 8:
                start_darkflow(Model(i+2))
            break
