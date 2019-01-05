import torch
import shutil
import os

def save_checkpoint(state, is_best, file_path='./'):
    file_name = os.path.join(file_path, 'checkpoint.pth.tar')
    torch.save(state, file_name)
    if is_best:
        best_file_name = os.path.join(file_path, 'model_best.pth.tar')
        shutil.copyfile(file_name, best_file_name)

def load_checkpoint(is_best, file_path='./'):
    checkpoint = None
    if is_best:
        chkpt_file = os.path.join(file_path, 'model_best.pth.tar')
    else:
        chkpt_file = os.path.join(file_path, 'checkpoint.pth.tar')

    if os.path.isfile(chkpt_file):
        print("=> loading checkpoint '{}'".format(chkpt_file))
        checkpoint = torch.load(chkpt_file)
    else:
        print("=> no checkpoint found at '{}'".format(chkpt_file))
    return checkpoint