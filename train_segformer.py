
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from core.models.segformer import SegFormer
from core.models.segment_model import DeepSegmenter
from core.dataset.cityscapes import CityscapesCustom
from core.dataset.oxfordpets import OxfordPetsCustom
from core.metrics import SegMetrics
from core.trainer import ImgSemSegTrainer


def train(args):
    # Define the transformations
    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    if args.dataset == "oxford":
        dataset_path = os.path.join(os.path.dirname(__file__), 'data\\oxfordpets')
        #Download oxforpets dataset if not already downloaded
        download = False if os.path.exists(os.path.join(dataset_path, 'oxford-iiit-pet')) else True
        train_data = OxfordPetsCustom(root=dataset_path,
                                split="trainval",
                                target_types='segmentation', 
                                transform=train_transform,
                                target_transform=train_transform2,
                                download=download)

        val_data = OxfordPetsCustom(root=dataset_path,
                                split="test",
                                target_types='segmentation', 
                                transform=val_transform,
                                target_transform=val_transform2,
                                download=download)
    if args.dataset == "city":
        #Expects the cityscapes dataset to be in the data folder, you have to add this manually
        dataset_path = os.path.join(os.path.dirname(__file__), 'data\\cityscapes')
        train_data = CityscapesCustom(root=dataset_path,
                                split="train",
                                mode="fine",
                                target_type='semantic', 
                                transform=train_transform,
                                target_transform=train_transform2)
        val_data = CityscapesCustom(root=dataset_path, 
                                split="val",
                                mode="fine",
                                target_type='semantic', 
                                transform=val_transform,
                                target_transform=val_transform2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSegmenter(SegFormer(num_classes=len(train_data.classes_seg)))

    # If you are in the fine-tuning phase:
    if args.dataset == 'oxford':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)  # ignore the void class

    if args.load_model:
        model_load_dir = Path("saved_models", args.load_dir) 
        model_load_dir.mkdir(exist_ok=True)
        model.load(model_load_dir, 'best')  # Loading the backbone model

    if args.freeze_weights and args.load_model:  # Freeze the weights of the backbone model
        for param in model.net.encoder.parameters():
            param.requires_grad = False
        params = model.net.decoder.parameters()
    else:
        params = model.parameters()

    model.to(device)
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, amsgrad=True)

    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2
    
    model_save_dir = Path("saved_models", args.save_dir)
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    trainer = ImgSemSegTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=16,
                    val_frequency = val_frequency,
                    run_name=args.run_name)

    trainer.train()
    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 
    

if __name__ == "__main__":
    # We will train the model in 4 different ways
    # For this we will change the arguments and call the train function different times

    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 40

    # Pretraining of the SegFormer on the cityscapes dataset
    args.dataset = "city"
    args.load_model = False
    args.load_dir = None
    args.freeze_weights = False
    args.lr = 0.001
    args.save_dir = 'segformer_pretraining'
    args.run_name = "SegFormer_pretraining"
    train(args)

    # Training the SegFormer from scratch on the oxfordpets dataset
    args.dataset = "oxford"
    args.run_name = "SegFormer_from-scratch"
    args.save_dir = 'segformer_from_scratch'
    args.num_epochs = 30
    train(args)

    # Fine-tuning the pre-trained SegFormer model (on the cityscape dataset) on the oxfordpets dataset
    args.load_model = True
    args.load_dir = 'segformer_pretraining'
    args.run_name = 'segformer_finetuning'
    args.save_dir = 'segformer_finetuning'
    train(args)

    # Freezing the weights of the backbone model and fine-tuning the SegFormer on the oxfordpets dataset
    args.freeze_weights = True
    args.run_name = 'segformer_freeze_weights'
    args.save_dir = 'segformer_freeze_weights'
    train(args)

    # Freezing the weights of the backbone model and fine-tuning the SegFormer on the oxfordpets dataset with a learning rate of 0.0005
    args.lr = 0.0005
    args.run_name = 'segformer_freeze_weights_lr_halfed'
    args.save_dir = 'segformer_freeze_weights_lr_halfed'
    train(args)

    # Fine-tuning the SegFormer on the oxfordpets dataset with a learning rate of 0.0005
    args.freeze_weights = False
    args.run_name = 'segformer_finetuning_lr_halfed'
    args.save_dir = 'segformer_finetuning_lr_halfed'
    train(args)
