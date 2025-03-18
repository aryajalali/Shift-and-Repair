import torch
from tqdm import tqdm

from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from heads import get_classification_head
from modeling import ImageClassifier, SurgeryWrapper
from repair import ShiftWrapper
from task_vectors import TaskVector
from ties_merging_utils import *
from variables_and_paths import get_zeroshot_path, get_finetuned_path


from methods import *



def get_finetuned_encoder(dataset, args):

    pretrained_checkpoint = get_zeroshot_path(args.model_location, args.model)
    task_vector = TaskVector(pretrained_checkpoint, get_finetuned_path(args.model_location, dataset, args.model))

    ft_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef= 1.0)

    return ft_encoder


def get_encoder(args):

    encoder, individual_encoders = None, None

    # Task Arithmetic
    if args.method.name == "sum":
        encoder = taskArithmetic(args)

    # EMR merging
    elif args.method.name == "emr":
        encoder, individual_encoders = EMR(args)

    # TIES merging
    elif args.method.name == "ties":
        encoder = ties(args)

    elif args.method.name == "tw_adamerging":
        encoder =  taskwise_adamerging(args)

    elif args.method.name == "lw_adamerging":
        encoder = layerwise_adamerging(args)

    return encoder, individual_encoders


         

def feature_shifting(merged_encoder, dataset_name ,args, finetuned_encoder = None):
        
        if finetuned_encoder is None:
            finetuned_encoder = get_finetuned_encoder(dataset_name, args)

        ft_model = ImageClassifier(finetuned_encoder, get_classification_head(args, dataset_name)).to(args.device)
 
        merged_model = ImageClassifier(merged_encoder, get_classification_head(args, dataset_name)).to(args.device)

        shift_model = ShiftWrapper(finetuned_encoder, merged_encoder, get_classification_head(args, dataset_name), batch_shift = True, 
                                dataset = dataset_name, args = args).to(args.device)
        
        dataset = get_dataset(dataset_name, ft_model.val_preprocess, location=args.data_location, batch_size=args.batch_size)

        test_dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)

        accuracy = 0.0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for data in tqdm(test_dataloader, total=len(test_dataloader), desc=f"Testing {dataset_name} using mask"):
                data = maybe_dictionarize(data)
                x = data["images"].to(args.device)
                y = data["labels"].to(args.device)
            
                logits = shift_model(x)
                
                predictions = logits.argmax(dim=1)
                correct_predictions += (predictions == y).sum().item()

                total_samples += y.size(0)

        accuracy = correct_predictions / total_samples


        print(f"Accuracy: {accuracy * 100:.2f}%")


def evaluate(model, dataloader, dataset_name, args, mode="Validation"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Evaluating {dataset_name} ({mode})", leave=False):
            data = maybe_dictionarize(data)
            x, y = data["images"].to(args.device), data["labels"].to(args.device)

            logits = model(x, return_features=False)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f"{dataset_name} {mode} Accuracy: {accuracy:.4f}")
    return accuracy


def repair_head(merged_encoder, dataset_name, args):
    """Repairs the classification head of the merged model using fine-tuned features."""

    iterations, eval_interval = 1000, 100
    mse_loss = torch.nn.MSELoss()

    ft_encoder = get_finetuned_encoder(dataset_name, args)

    ft_model = ImageClassifier(ft_encoder, get_classification_head(args, dataset_name)).to(args.device)
    merged_model = ImageClassifier(merged_encoder, get_classification_head(args, dataset_name)).to(args.device)

    if not dataset_name.endswith("Val"):
        dataset_name += "Val"

    merged_model.freeze_head()
    merged_model.freeze_encoder()
    ft_model.eval()

    for param in ft_model.parameters():
        param.requires_grad = False

    for param in merged_model.parameters():
        param.requires_grad = False

    merged_model.classification_head.mask = torch.nn.Parameter(torch.ones_like(merged_model.classification_head.weight))
    merged_model.classification_head.mask.requires_grad_(True)

    dataset = get_dataset(dataset_name, merged_model.val_preprocess, args.data_location, args.batch_size)
    train_loader = get_dataloader(dataset, is_train=True, args=args)
    val_loader = get_dataloader(dataset, is_train=False, args=args)

    optimizer = torch.optim.Adam(
        [p for p in merged_model.parameters() if p.requires_grad],
        lr=1e-2, betas=(0.9, 0.999), weight_decay=0.
    )
    

    best_acc = 0

    print(f"Number of trainable parameters: {sum(p.numel() for p in merged_model.parameters() if p.requires_grad)}")

    for i in tqdm(range(iterations), desc="Training Head Repair"):
        merged_model.train()
        for data in train_loader:
            data = maybe_dictionarize(data)
            x = data["images"].to(args.device)

            # logits = merged_model(x)
            features = merged_model.image_encoder(x).to(args.device)
            features = features.detach()
            logits = merged_model.classification_head(features).to(args.device)
            ft_logits = ft_model(x).detach().to(args.device)

            loss = mse_loss(logits, ft_logits).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break

        if (i + 1) % eval_interval == 0 or i == iterations - 1:
            val_accuracy = evaluate(merged_model,val_loader , dataset_name, args, mode="Validation")
            if val_accuracy > best_acc:
                print(f"New best accuracy: {val_accuracy}")
                best_acc = val_accuracy
                best_model = merged_model.state_dict()

    merged_model.load_state_dict(best_model)

    print("Testing...")
    test_dataset = get_dataset(dataset_name[:-3], merged_model.val_preprocess, args.data_location, args.batch_size)
    test_loader = get_dataloader(test_dataset, is_train=False, args=args)

    return evaluate(merged_model,test_loader ,dataset_name[:-3], args, mode="Test")



def surgery(merged_encoder, dataset, args):

    ft_encoder = get_finetuned_encoder(dataset, args)
    
    merged_model = ImageClassifier(merged_encoder, get_classification_head(args, dataset)).to(args.device)
    ft_model = ImageClassifier(ft_encoder, get_classification_head(args, dataset)).to(args.device)

    dataset_name = dataset if dataset.endswith("Val") else dataset + "Val"

    model_with_surgery = SurgeryWrapper(merged_model).to(args.device)
    model_with_surgery.freeze_base_model()

    ft_model.freeze_head()
    ft_model.freeze_encoder()
    ft_model.eval()

    dataset = get_dataset(dataset_name, merged_model.val_preprocess, args.data_location, args.batch_size)
    train_loader = get_dataloader(dataset, is_train=True, args=args)
    val_loader = get_dataloader(dataset, is_train=False, args=args)

    l1_loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_with_surgery.parameters()), 
        lr=1e-3, betas=(0.9, 0.999), weight_decay=0.
    )

    iterations, eval_interval = 1000, 100
    for i in tqdm(range(iterations), desc="Training Surgery Module"):
        model_with_surgery.train()
        for data in train_loader:
            data = maybe_dictionarize(data)
            x, y = data["images"].to(args.device), data["labels"].to(args.device)

            _, adapter_features = model_with_surgery(x, return_features=True)
            _, ft_features = ft_model(x, return_features=True)

            loss = l1_loss(adapter_features, ft_features.detach()).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break 

        if (i + 1) % eval_interval == 0 or i == iterations - 1:
            evaluate(model_with_surgery, val_loader, dataset_name, args, mode="Validation")

    print("Testing...")
    test_dataset = get_dataset(dataset_name[:-3], merged_model.val_preprocess, args.data_location, args.batch_size)
    test_loader = get_dataloader(test_dataset, is_train=False, args=args)

    evaluate(model_with_surgery, test_loader, dataset_name[:-3], args, mode="Test")

