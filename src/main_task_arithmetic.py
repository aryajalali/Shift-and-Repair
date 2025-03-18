import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import sys
import torch
sys.path.append('/Samira/AdaMerging/')
torch.manual_seed(0)

from tqdm import tqdm

from task_vectors import TaskVector
c
from args import parse_arguments

from datasets.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head
from modeling import ImageClassifier
from src.repair_and_shift_utils import *

from datasets.registry import get_dataset

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

exam_datasets = ['GTSRB', 'MNIST', 'SUN397', 'Cars', 'SVHN', 'RESISC45', 'EuroSAT', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
model = 'ViT-B-32'
args = parse_arguments()
args.data_location = '/Samira/AdaMerging//data'
args.model = model
args.save = '/Samira/AdaMerging/checkpoints/' + model
args.logs_path = '/Samira/AdaMerging//logs/' + model
pretrained_checkpoint = '/Samira/AdaMerging//checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))

task_vectors = [
    TaskVector(pretrained_checkpoint, '/Samira/AdaMerging//checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets
]

task_vector_sum = sum(task_vectors)

scaling_coef_ = 0.3

image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef_)
log.info('*'*20 + 'scaling_coef:' + str(scaling_coef_) + '*'*20)

accs = []
for idx, dataset in enumerate(exam_datasets):
    finetune_encoder = task_vectors[idx].apply_to(pretrained_checkpoint, 1.0)
    # metrics = eval_single_dataset(finetune_encoder, dataset, args)
    # feature_normalization(finetune_encoder, image_encoder, dataset, args)

    # repair_head(image_encoder, finetune_encoder, dataset, args)

    feature_debug(finetune_encoder, image_encoder, dataset, args)

    # surgery(image_encoder, finetune_encoder, dataset, args)

    # log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    # accs.append(metrics.get('top1')*100)
# log.info('Avg ACC:' + str(np.mean(accs)) + '%')
