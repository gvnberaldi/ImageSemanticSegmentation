
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torchvision.models.segmentation import fcn_resnet50

from core.models.segment_model import DeepSegmenter
from core.dataset.oxfordpets import  OxfordPetsCustom
from core.metrics import SegMetrics
from core.trainer import ImgSemSegTrainer


from torch.profiler import profile, record_function, ProfilerActivity

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


    dataset_path = os.path.join(os.path.dirname(__file__), 'data','oxfordpets')

    #Dpwnload the dataset if not already downloaded
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training from scratch
    model = DeepSegmenter(fcn_resnet50(weights=None, num_classes=3))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2

    model_save_dir = Path("saved_models/from_scratch")
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
                    batch_size=64,
                    val_frequency = val_frequency,
                    run_name="FCN_from-scratch")
    
    trainer.train()

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose()

    # Pretrained weights
    model = DeepSegmenter(fcn_resnet50(weights_backbone='DEFAULT', num_classes=3)) # Use default weights for backbone, these get automatically downloaded
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2

    model_save_dir = Path("saved_models/pretrained")
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
                    batch_size=64,
                    val_frequency = val_frequency,
                    run_name = "FCN")
    
    trainer.train()

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose()

    # Freeze backbone
    model = DeepSegmenter(fcn_resnet50(weights_backbone='DEFAULT', num_classes=3))
    model.to(device)
    for param in model.net.backbone.parameters():
        param.requires_grad = False # Set requires_grad to False for all parameters in the backbone to freeze them

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2

    model_save_dir = Path("saved_models/freezed_backbone")
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
                    batch_size=64,
                    val_frequency = val_frequency,
                    run_name="FCN_freeze_backbone")
    
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    trainer.train()

    # prof.export_chrome_trace("trace.json")

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 30

    train(args)