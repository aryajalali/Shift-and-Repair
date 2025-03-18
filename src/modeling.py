import torch

import open_clip
from math import sqrt
from math import log
import utils

from utils import make_functional, load_weights

from variables_and_paths import OPENCLIP_CACHEDIR, CACHEDIR


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        if '__pretrained__' in args.model:
            name, pretrained = args.model.split('__pretrained__')
        else:
            name = args.model
            pretrained = 'openai'
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=OPENCLIP_CACHEDIR)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)
    
    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f'Loading image encoder from {filename}')
        state_dict = torch.load(filename, map_location='cpu', weights_only=False)
        model = cls(model_name)
        model.load_state_dict(state_dict)
        return model



class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    
    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)

        # check if mask exists 
        if hasattr(self, "mask"):
            masked_weight = self.weight * self.mask
            return torch.nn.functional.linear(inputs, masked_weight, self.bias)

        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)
    

class SurgeryModule(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )

        self.down_proj = torch.nn.Linear(input_dim, hidden_dim, bias = False)
        self.up_proj = torch.nn.Linear(hidden_dim, input_dim, bias = False)

        torch.nn.init.kaiming_uniform_(self.down_proj.weight, a=sqrt(5))
        torch.nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        x = self.down_proj(x)
        x = torch.nn.functional.relu(x)
        x = self.up_proj(x)
        return x
    

class SurgeryWrapper(torch.nn.Module):
    def __init__(self, base_model, adapter_hidden_dim=16):
        super().__init__()
        self.base_model = base_model
        self.adapter = SurgeryModule(
            input_dim=base_model.image_encoder.model.visual.output_dim,
            # hidden_dim=adapter_hidden_dim
        )

    def forward(self, x, return_features=False):
        features = self.base_model.image_encoder(x)
        adjustments = self.adapter(features)
        if return_features:
            return self.base_model.classification_head(features - adjustments), features - adjustments
        return self.base_model.classification_head(features - adjustments)

    def freeze_base_model(self):
        """Freezes everything except the adapter."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.adapter.parameters():
            param.requires_grad = True

    def save(self, filename):
        print(f"Saving adapter-wrapped model to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, base_model, filename):
        print(f"Loading adapter-wrapped model from {filename}")
        wrapper = cls(base_model)
        wrapper.adapter.load_state_dict(torch.load(filename, map_location="cuda"))
        return wrapper


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def unfreeze_head(self):
        self.classification_head.weight.requires_grad_(True)
        self.classification_head.bias.requires_grad_(True)

    def freeze_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, inputs, return_features=False):

        features = self.image_encoder(inputs)  
        outputs = self.classification_head(features)

        if return_features:
            return outputs, features
        return outputs

    def __call__(self, inputs, return_features=False):
        return self.forward(inputs, return_features)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)




'''AdaMerging stuff'''


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features

class TW_AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, args):
        super(TW_AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(1, 1)
        self.args = args
        prior = 0.3
        rlambdas = torch.ones(1, len(paramslist)-1) * prior  # (1 * tasks)
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        # self.classifier = []
        # for dataset_name in exam_datasets:
        #     classification_head = get_classification_head(args, dataset_name)
        #     layer_name = 'classifier_{}'.format(dataset_name)
        #     self.add_module(layer_name, classification_head.to(args.device))
        #     self.classifier.append(layer_name)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    # def get_classification_head(self, dataset_name):
    #     layer_name = 'classifier_{}'.format(dataset_name)
    #     classification_head = getattr(self, layer_name)
    #     return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model
    
    def set_weights(self, alph):
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)

        return out
    

class LW_AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets):
        super(LW_AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
        prior = 0.3
        rlambdas = torch.ones(len(paramslist[0]), len(paramslist)-1) * prior  # (1 * tasks)
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        # self.classifier = []
        # for dataset_name in exam_datasets:
        #     classification_head = get_classification_head(args, dataset_name)
        #     layer_name = 'classifier_{}'.format(dataset_name)
        #     self.add_module(layer_name, classification_head.to(args.device))
        #     self.classifier.append(layer_name)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    # def get_classification_head(self, dataset_name):
    #     layer_name = 'classifier_{}'.format(dataset_name)
    #     classification_head = getattr(self, layer_name)
    #     return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model
    

    def set_weights(self, alph):
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model


    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)
        return out



class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, mode="task-wise", prior=0.3):
        """
        Unified class for TW_AdaMerging (Task-wise) and LW_AdaMerging (Layer-wise).

        Args:
            paramslist (list): List of model parameters from different tasks.
            model (torch.nn.Module): Base model to merge weights into.
            names (list): Names of layers whose parameters are merged.
            mode (str): "task-wise" for TW_AdaMerging, "layer-wise" for LW_AdaMerging.
            prior (float): Initial value for lambda parameters.
        """
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.mode = mode

        if mode == "task-wise":
            self.pretrain_lambdas = torch.ones(1, 1)
            rlambdas = torch.ones(1, len(paramslist) - 1) * prior  # (1 × tasks)
        elif mode == "layer-wise":
            self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
            rlambdas = torch.ones(len(paramslist[0]), len(paramslist) - 1) * prior  # (layers × tasks)
        else:
            raise ValueError("Mode must be 'task-wise' or 'layer-wise'")

        self.lambdas_raw = torch.nn.Parameter(rlambdas)

    def lambdas(self):
        """Returns clamped lambda values."""
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        return torch.cat((self.pretrain_lambdas, task_lambdas), 1)

    def collect_trainable_params(self):
        """Returns the parameters that can be trained."""
        return [self.lambdas_raw]

    def get_image_encoder(self):
        """Computes and loads merged parameters into the model."""
        alph = self.lambdas()
        params = self.compute_merged_params(alph)
        load_weights(self.model, self.names, params)
        return self.model

    def set_weights(self, alph):
        """Manually sets the merged weights using a given lambda."""
        params = self.compute_merged_params(alph)
        load_weights(self.model, self.names, params)
        return self.model

    def compute_merged_params(self, alph):
        """Computes the merged parameters based on the current lambda values."""
        if self.mode == "task-wise":
            params = tuple(
                sum(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))
                for p in zip(*self.paramslist)
            )
        else:  # "layer-wise"
            params = tuple(
                sum(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))
                for j, p in enumerate(zip(*self.paramslist))
            )
        return tuple(p.cuda(0) for p in params)

    def forward(self, inp, dataset_name):
        """Forward pass with merged weights and classification head."""
        alph = self.lambdas()
        params = self.compute_merged_params(alph)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = f'classifier_{dataset_name}'
        classification_head = getattr(self, layer_name)
        return classification_head(feature)