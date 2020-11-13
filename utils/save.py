import os
import torch
from tqdm import tqdm
import numpy as np
from skimage.io import imsave


def save_samples(folder_to_save, num_images_to_save, generator, nz, device):
    full_path = os.path.join(folder_to_save, 'generated')
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    for i in tqdm(range(num_images_to_save)):
        gen_z = torch.randn(1, nz, 1, 1, device=device)
        generated_image = generator(gen_z)
        image = generated_image.squeeze(0).squeeze(0)
        image = image.cpu().clone().detach().numpy()
        image = ((image + 1) * 255 / 2).astype(np.uint8)
        img_label = os.path.join(full_path, f'gen_{i}.png')
        imsave(img_label, image)
