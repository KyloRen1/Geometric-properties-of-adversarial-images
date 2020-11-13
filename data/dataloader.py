import os
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_dataloader(dataset, batch_size, data_path, resize=64):
    os.makedirs(data_path, exist_ok=True)
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, ), std=(0.5,))
        ])
        dataset = torchvision.datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
    else:
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.CelebA(
            root=data_path,
            split='train',
            download=True,
            transform=transform
        )

    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


class ImageDataset(Dataset):
    def __init__(self, img_directory, transform=None):
        self.images = []
        self.labels = []
        self.image_names = []
        self.n_images = 10000
        for idx, directory in enumerate(img_directory):
            label = 'fake' if idx == 0 else 'real'
            for idx, image_name in enumerate(os.listdir(directory)):
                if idx > self.n_images:
                    break
                img = Image.open(os.path.join(directory, image_name))
                if transform is not None:
                    img = transform(img)
                self.images.append(img)
                self.labels.append(label)
                self.image_names.append(image_name)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        image_name = self.image_names[index]
        return (img, label, image_name)

    def __len__(self):
        return len(self.images)


def get_dataloader_experiment(
        real_img_path,
        generated_img_path,
        resize=64,
        batch_size=1):

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataset = ImageDataset(
        img_directory=[generated_img_path, real_img_path],
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
