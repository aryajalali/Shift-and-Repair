import os
import sys
import torch
import hydra
import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from eval import eval_single_dataset


from repair_and_shift_utils import get_encoder, surgery, repair_head, feature_shifting

# Set visible GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add external paths
sys.path.append('/Samira/AdaMerging/')

# Define constants
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
MODELS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]
ALL_DATASETS = [
    "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN", "SUN397",
    "STL10", "OxfordIIITPet", "Flowers102", "CIFAR100", "PCAM", "FER2013",
    "CIFAR10", "Food101", "FashionMNIST", "RenderedSST2", "EMNIST", "KMNIST",
]


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:

    torch.manual_seed(0)

    print(f"Using model: {cfg.model}")

    cfg.DATASETS = ALL_DATASETS[:cfg.num_tasks] if not cfg.DATASETS else cfg.DATASETS
    cfg.num_tasks = len(cfg.DATASETS)

    cfg.data_location = Path(cfg.data_location).expanduser().as_posix()

    OmegaConf.set_struct(cfg, True)

    merged_encoder, individual_encoders = get_encoder(cfg)

    accuracies = {}

    for idx, dataset_name in enumerate(cfg.DATASETS):

        if individual_encoders is not None:
            encoder = individual_encoders[idx]
        else:
            encoder = merged_encoder

        if dataset_name !=  "DTD":
            continue

        # eval_single_dataset(encoder, dataset_name, cfg)
        # surgery(merged_encoder, dataset_name, cfg)

        # feature_shifting(sum_encoder, dataset_name, cfg, finetuned_encoder=individual_encoders[idx])

        accuracies[dataset_name] = repair_head(encoder, dataset_name, cfg)

        # log it after each dataset (save it in a json file)
        with open('accuracies.json', 'w') as f:
            json.dump(accuracies, f)





if __name__ == "__main__":
    my_app()
