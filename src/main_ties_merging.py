import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import sys
import pickle
import torch
sys.path.append('/Samira/AdaMerging/')

from eval import eval_single_dataset
from args import parse_arguments

from task_vectors import TaskVector
from modeling import ImageClassifier,  ImageEncoder

from src.repair_and_shift_utils import feature_normalization, repair_head


torch.manual_seed(0)


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

exam_datasets = ['DTD', 'SUN397', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'Cars'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD

# exam_datasets = ['Cars', 'DTD']

model = 'ViT-B-32'
args = parse_arguments()
args.data_location = '/Samira/AdaMerging//data'
args.model = model
args.save = '/Samira/AdaMerging/checkpoints/' + model
args.logs_path = '/Samira/AdaMerging//logs/' + model
pretrained_checkpoint = '/Samira/AdaMerging//checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))

from ties_merging_utils import *
ft_checks = [torch.load('/Samira/AdaMerging/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt'
                        , weights_only=False, map_location='cpu').state_dict() for dataset_name in exam_datasets]

ptm_check = torch.load(pretrained_checkpoint, weights_only=False, map_location='cpu').state_dict()
check_parameterNamesMatch(ft_checks + [ptm_check])

remove_keys = []
print(f"Flattening out Checkpoints")
flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

tv_flat_checks = flat_ft - flat_ptm
assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i])for i in range(len(ft_checks))])


K = 20
merge_func = "dis-sum"
scaling_coef_ = 0.3

merged_tv = ties_merging(tv_flat_checks, reset_thresh=K, merge_func=merge_func,)
merged_check = flat_ptm + scaling_coef_ * merged_tv
merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)

image_encoder = torch.load(pretrained_checkpoint, weights_only=False, map_location='cpu')
image_encoder.load_state_dict(merged_state_dict, strict=False)

task_vectors = [
    TaskVector(pretrained_checkpoint, '/Samira/AdaMerging//checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets
]

Total_ACC = 0.
for idx, dataset in enumerate(exam_datasets):

    finetune_encoder = task_vectors[idx].apply_to(pretrained_checkpoint, 1.0)

    # feature_normalization(finetune_encoder, image_encoder, dataset, args)

    repair_head(image_encoder, finetune_encoder, dataset, args)

    # metrics = eval_single_dataset(image_encoder, dataset, args)
    # Total_ACC += metrics['top1']
    # log.info(str(dataset) + ':' + str(metrics))

# log.info('Final: ' + 'Avg ACC:' + str(Total_ACC / len(exam_datasets)))
