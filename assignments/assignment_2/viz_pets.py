import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
import os
import matplotlib.pyplot as plt
import numpy as np
from train import OxfordPetsCustom


def imshow(img, filename='img/test.png'):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave(filename,np.transpose(npimg, (1, 2, 0)))


if __name__ == '__main__':
    image_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True),
                                  v2.Resize(size=(300, 300), interpolation=v2.InterpolationMode.NEAREST),
                                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    label_transform = v2.Compose([v2.ToImage(),
                                  v2.ToDtype(torch.long, scale=False),
                                  v2.Resize(size=(300, 300), interpolation=v2.InterpolationMode.NEAREST)])  # ,

    # OxfordPets dataset
    print('TESTING OXFORDPETS DATASET')
    dataset_path = os.path.join(os.path.dirname(__file__), 'data\\oxfordpets')
    download = False if os.path.exists(os.path.join(dataset_path, 'oxford-iiit-pet')) else True

    train_data = OxfordPetsCustom(root=dataset_path,
                                  split="trainval",
                                  target_types='segmentation',
                                  transform=image_transform,
                                  target_transform=label_transform,
                                  download=download)

    train_data_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=2)

    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)

    images_plot = torchvision.utils.make_grid(images, nrow=4)
    labels_plot = torchvision.utils.make_grid((labels-1)/2, nrow=4)#.to(torch.uint8)

    # show/plot images
    imshow(images_plot, filename="img/input_test_pets.png")
    imshow(labels_plot,filename="img/seg_mask_test_pets.png")

