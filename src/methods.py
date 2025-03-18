from typing import List, Tuple, Optional
import torch
import copy

from task_vectors import TaskVector
from variables_and_paths import get_zeroshot_path, get_finetuned_path
from modeling import ModelWrapper, AdaMerging
from utils import make_functional
from ties_merging_utils import (
    check_parameterNamesMatch,
    state_dict_to_vector,
    vector_to_state_dict,
    check_state_dicts_equal,
    ties_merging
)
from merging_cofficient import get_merging_cofficients


def taskArithmetic(args) -> torch.nn.Module:
    """
    Perform task arithmetic merging of multiple task vectors.
    
    Args:
        args: Arguments containing model location, model name, and datasets
        
    Returns:
        Merged model encoder
    """
    scaling_coeff = 0.3
    pretrained_checkpoint = get_zeroshot_path(args.model_location, args.model)
    task_vectors = [
        TaskVector(pretrained_checkpoint, get_finetuned_path(args.model_location, dataset_name, args.model))
        for dataset_name in args.DATASETS
    ]

    task_vector_sum = sum(task_vectors)
    return task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coeff)


def ties(args) -> torch.nn.Module:
    """
    Perform TIES merging of multiple fine-tuned models.
    
    Args:
        args: Arguments containing model location, model name, and datasets
        
    Returns:
        Merged model encoder
    """
    pretrained_checkpoint = get_zeroshot_path(args.model_location, args.model)
    
    # Load checkpoints
    ft_checks = [
        torch.load(f"{args.model_location}/{args.model}/{dataset_name}/finetuned.pt", 
                  weights_only=False, map_location='cpu').state_dict() 
        for dataset_name in args.DATASETS
    ]
    ptm_check = torch.load(pretrained_checkpoint, weights_only=False, map_location='cpu').state_dict()
    
    # Validate parameter names
    check_parameterNamesMatch(ft_checks + [ptm_check])

    # Flatten checkpoints
    remove_keys = []
    print("Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    # Calculate task vectors
    tv_flat_checks = flat_ft - flat_ptm
    
    # Validate state dicts
    assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
    assert all([
        check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i])
        for i in range(len(ft_checks))
    ])

    # Perform TIES merging
    K = 20
    merge_func = "dis-sum"
    scaling_coef_ = 0.3

    merged_tv = ties_merging(tv_flat_checks, reset_thresh=K, merge_func=merge_func)
    merged_check = flat_ptm + scaling_coef_ * merged_tv
    merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)

    # Load and return merged model
    image_encoder = torch.load(pretrained_checkpoint, weights_only=False, map_location='cpu')
    image_encoder.load_state_dict(merged_state_dict, strict=False)
    return image_encoder


def _prepare_adamerging(args) -> Tuple[ModelWrapper, List[Tuple[torch.Tensor, ...]], List[str]]:
    """Helper function to prepare model and parameters for AdaMerging."""
    pretrained_checkpoint = get_zeroshot_path(args.model_location, args.model)
    task_vectors = [
        TaskVector(pretrained_checkpoint, get_finetuned_path(args.model_location, dataset_name, args.model))
        for dataset_name in args.DATASETS
    ]
    
    pretrained_model = torch.load(pretrained_checkpoint, weights_only=False, map_location='cpu')
    pretrained_model_dic = pretrained_model.state_dict()

    model = ModelWrapper(pretrained_model, args.DATASETS)
    model = model.to(args.device)
    _, names = make_functional(model)

    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())]
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items()) for tv in task_vectors]
    torch.cuda.empty_cache()

    return model, paramslist, names


def taskwise_adamerging(args) -> torch.nn.Module:
    """
    Perform task-wise AdaMerging of multiple models.
    
    Args:
        args: Arguments containing model location, model name, and datasets
        
    Returns:
        Merged model encoder
    """
    model, paramslist, names = _prepare_adamerging(args)
    adamerging_mtl_model = AdaMerging(paramslist, model, names, mode="task-wise")
    ralphas = get_merging_cofficients(args.method.name, args.model)
    return adamerging_mtl_model.set_weights(torch.Tensor(ralphas)).model


def layerwise_adamerging(args) -> torch.nn.Module:
    """
    Perform layer-wise AdaMerging of multiple models.
    
    Args:
        args: Arguments containing model location, model name, and datasets
        
    Returns:
        Merged model encoder
    """
    model, paramslist, names = _prepare_adamerging(args)
    adamerging_mtl_model = AdaMerging(paramslist, model, names, mode="layer-wise")
    ralphas = get_merging_cofficients(args.method.name, args.model)
    return adamerging_mtl_model.set_weights(torch.Tensor(ralphas)).model


def EMR(args) -> Tuple[torch.nn.Module, List[torch.nn.Module]]:
    """
    Perform EMR (Ensemble Model Reduction) merging of multiple models.
    
    Args:
        args: Arguments containing model location, model name, and datasets
        
    Returns:
        Tuple containing:
        - Merged model encoder
        - List of individual EMR models
    """
    pretrained_checkpoint = get_zeroshot_path(args.model_location, args.model)
    
    # Load checkpoints
    ft_checks = [
        torch.load(f"{args.model_location}/{args.model}/{dataset_name}/finetuned.pt",
                  weights_only=False, map_location='cpu').state_dict()
        for dataset_name in args.DATASETS
    ]
    ptm_check = torch.load(pretrained_checkpoint, weights_only=False, map_location='cpu').state_dict()
    
    # Validate parameter names
    check_parameterNamesMatch(ft_checks + [ptm_check])

    # Flatten checkpoints
    remove_keys = []
    print("Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    # Calculate task vectors
    tv_flat_checks = flat_ft - flat_ptm
    
    # Validate state dicts
    assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
    assert all([
        check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i])
        for i in range(len(ft_checks))
    ])

    # Calculate unified task vector
    y_uni = torch.sign(torch.sum(tv_flat_checks, dim=0))
    e_uni = torch.max(tv_flat_checks * y_uni, dim=0).values
    tv_uni = e_uni * y_uni

    # Process individual task vectors
    emr_tvs = []
    for tv in tv_flat_checks:
        tv_mask = (tv * tv_uni > 0)
        tv_lambda = torch.sum(torch.abs(tv)) / torch.sum(torch.abs(tv_mask * tv_uni))

        emr_tv = vector_to_state_dict(tv_lambda * (tv_mask * tv_uni) + flat_ptm, ptm_check, remove_keys=remove_keys)
        encoder = torch.load(pretrained_checkpoint, weights_only=False, map_location='cpu')
        encoder.load_state_dict(emr_tv, strict=False)
        emr_tvs.append(copy.deepcopy(encoder))

    # Create final merged model
    merged_state_dict = vector_to_state_dict(tv_uni + flat_ptm, ptm_check, remove_keys=remove_keys)
    encoder = torch.load(pretrained_checkpoint, weights_only=False, map_location='cpu')
    encoder.load_state_dict(merged_state_dict, strict=False)

    return encoder, emr_tvs

