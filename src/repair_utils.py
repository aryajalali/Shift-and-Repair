from heads import get_classification_head
from eval import get_dataset, get_dataloader, maybe_dictionarize
from modeling import ImageClassifier

from tqdm import tqdm
import torch


def collect_stats(finetuned_encoder, merged_encoder, dataset, args):
    device = args.device

    ft_model = ImageClassifier(finetuned_encoder, get_classification_head(args, dataset)).to(device)
    merged_model = ImageClassifier(merged_encoder, get_classification_head(args, dataset)).to(device)

    dataset_name = dataset if dataset.endswith("Val") else dataset + "Val"

    dataset = get_dataset(dataset_name, ft_model.val_preprocess, location=args.data_location, batch_size=args.batch_size)
    val_dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)

    all_features, all_merged_features = [], []

    with torch.no_grad():
        for data in tqdm(val_dataloader, desc=f"Collecting features for {dataset_name}"):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)

            all_features.append(ft_model(x, return_features=True)[1])
            all_merged_features.append(merged_model(x, return_features=True)[1])

    all_features = torch.cat(all_features, dim=0)
    all_merged_features = torch.cat(all_merged_features, dim=0)

    stats = {
        "mean": all_features.mean(dim=0),
        "var": all_features.var(dim=0),
        "merged_mean": all_merged_features.mean(dim=0),
        "merged_var": all_merged_features.var(dim=0),
    }

    return stats
