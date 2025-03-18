from modeling import *
from repair_utils import collect_stats


import torch
import torch.nn as nn



# Have two models (can be used to shift either at the last layer or at each layer)
class ShiftWrapper(torch.nn.Module):

    def __init__(self, finetuned_encoder, merged_encoder, classification_head, batch_shift = True, dataset = None, args = None):


        super().__init__()

        self.finetuned_encoder = finetuned_encoder
        self.merged_encoder = merged_encoder
        self.classification_head = classification_head
        self.batch_shift = batch_shift
        self.args = args


        if not self.batch_shift:
            assert dataset is not None
            self.stats = collect_stats(self.finetuned_encoder, self.merged_encoder, dataset, args)

        if self.merged_encoder is not None:
            self.train_preprocess = self.merged_encoder.train_preprocess
            self.val_preprocess = self.merged_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def unfreeze_head(self):
        self.classification_head.weight.requires_grad_(True)
        self.classification_head.bias.requires_grad_(True)

    def freeze_encoder(self):
        for param in self.merged_encoder.parameters():
            param.requires_grad = False

    def forward(self, inputs):

        if self.batch_shift:
            ft_features = self.finetuned_encoder(inputs)
            features = self.merged_encoder(inputs)
            features = (features - features.mean(dim = 0)) / features.var(dim = 0) * \
            ft_features.var(dim = 0) + ft_features.mean(dim = 0)
        else:
            features = self.merged_encoder(inputs)
            features = (features - self.stats["merged_mean"]) / self.stats["merged_var"] * self.stats["var"] + self.stats["mean"]
        
        return self.classification_head(features)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)