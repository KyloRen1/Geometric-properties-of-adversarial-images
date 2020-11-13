import argparse
import torch.optim as optim
from data import dataloader
from models import dcgan
from utils import save
import matplotlib.pyplot as plt
import os
import torch
from torch import nn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'celeba'], type=str
                        )
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--nz', default=100, type=int,
                        help='shape of generative vector'
                        )
    parser.add_argument(
        '--feature_map',
        default=64,
        type=int,
        help='size of feature maps in discriminator and generator')
    parser.add_argument('--data_folder', default='data/train_data', type=str)
    parser.add_argument('--num_samples_to_save', default=10000, type=int)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay'
                        )
    parser.add_argument('--plot_graphics', default=True, type=bool)
    return parser.parse_args()


def train_model(
        data_loader,
        discriminator,
        generator,
        num_epochs,
        criterion,
        discriminator_optimizer,
        generator_optimizer,
        nz,
        device,
        real_labels=1,
        fake_labels=0,
        plot_loss=True):

    generator_losses = []
    discriminator_losses = []

    print(' ===== Model training ===== ')
    for epoch in range(num_epochs):
        for idx, data in enumerate(data_loader):
            ####################################
            # Update Discriminator Network
            ####################################
            discriminator.zero_grad()
            real_samples = data[0]
            batch_size = real_samples.size(0)
            labels = torch.full(
                (batch_size, ), real_labels, device=device
            )

            d_output = discriminator(real_samples).view(-1)
            d_error_real = criterion(d_output, labels)
            d_error_real.backward()
            discriminator_x = d_output.mean().item()

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            g_output = generator(noise)
            labels.fill_(fake_labels)
            d_output = discriminator(g_output.detach()).view(-1)
            d_error_fake = criterion(d_output, labels)
            d_error_fake.backward()
            discriminator_generator_z1 = d_output.mean().item()

            error_discriminator = d_error_real + d_error_fake
            discriminator_optimizer.step()

            ####################################
            # Update Generator Network
            ####################################
            generator.zero_grad()
            labels.fill_(real_labels)
            d_output = discriminator(g_output).view(-1)
            error_generator = criterion(d_output, labels)
            error_generator.backward()
            discriminator_generator_z2 = d_output.mean().item()
            generator_optimizer.step()

            if idx % 100 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' %
                    (epoch +
                     1,
                     num_epochs,
                     idx,
                     len(data_loader),
                        error_discriminator.item(),
                        error_generator.item(),
                        discriminator_x,
                        discriminator_generator_z1,
                        discriminator_generator_z2))
            generator_losses.append(error_generator.item())
            discriminator_losses.append(error_discriminator.item())

    if plot_loss:
        plt.figure(figsize=(10, 5))
        plt.title('Generator and Discriminator loss')
        plt.plot(generator_losses, label='Generator')
        plt.plot(discriminator_losses, label='Discriminator')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    return generator, discriminator


def main():
    args = get_args()

    if args.dataset == 'mnist':
        n_channels = 1
    else:
        n_channels = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = dataloader.get_dataloader(
        args.dataset, args.batch_size, args.data_folder
    )
    generator, discriminator = dcgan.get_models(
        args.nz, n_channels, args.feature_map, device)

    loss_function = nn.BCELoss()
    optimizer_generator = optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta1),
        weight_decay=args.weight_decay
    )
    generator, discriminator = train_model(
        data_loader,
        discriminator,
        generator,
        args.epochs,
        loss_function,
        optimizer_discriminator,
        optimizer_generator,
        args.nz,
        device
    )
    print('Finished training')

    save.save_samples(
        args.data_folder,
        args.num_samples_to_save,
        generator,
        args.nz,
        device)
    print('Finished saving!')


if __name__ == '__main__':
    main()
