import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from data.dataloader import get_dataloader_experiment
from utils.math import cosine_similarity, centroid_point, parametric_line_equations, distance_between_two_points
######### load autoencoder model here
from models.autoencoder import MYAUTOENCODER


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'celeba'], type=str
                        )
    parser.add_argument('--data_path_real', type=str,
                        help='path to real data samples')
    parser.add_argument('--data_path_generated', type=str,
                        help='path to generated data sample')
    parser.add_argument('--autoencoder_weights_path', type=str)
    parser.add_argument('--number_of_samples', default=1000, type=int,
                        help='number of samples to perform experiment')
    parser.add_argument('--n_neighbours', default=10, type=int,
                        help='number of neighbour for each sample')
    parser.add_argument('--plot_graphics', default=True, type=bool)
    return parser.parse_args()

def construct_dataset(generated_image_path, real_image_path, autoencoder):
    dataloader = get_dataloader_experiment(
        real_image_path, generated_image_path)
    data = []
    for img, label, img_name in dataloader:
        encoded, _ = autoencoder(img)
        data.append((
            img_name[0],
            label[0],
            encoded.data.squeeze().view(-1).data.cpu().numpy()
        ))
    return data


def neightbours_search(data, target, n, cosine=True):
    neighbours = []
    for idx in range(len(data)):
        if data[idx][1] == 'real':
            if cosine:
                neighbours.append((
                    data[idx][0],
                    cosine_similarity(target[2], data[idx][2]),
                    idx
                ))
            else:
                neighbours.append((
                    data[idx][0],
                    euclidean_distance(target[2], data[idx][2]),
                    idx
                ))
    neighbours = sorted(
        neighbours, key=lambda x: x[1], reverse=cosine
    )[:(n + 1)]
    neighbours_values = []
    for neightbour in neighbours[1:]:
        neighbours_values.append(data[neightbour[2]])
    return neighbours_values


def distance_comparison_to_distribution(data, args, on_fake=True):
    if on_fake:
        target_idx = np.random.randint(0, args.number_of_samples * 10 - 1)
    else:
        target_idx = np.random.randint(
            args.number_of_samples * 10,
            args.number_of_samples * 10 + args.number_of_samples * 10)
    target = data[target_idx]

    # neighbours of a target
    neighbour_point_target = neightbours_search(
        data, target, args.n_neighbours, cosine=True)
    centroid_point_coordinates = centroid_point(
        [x[2] for x in neighbour_point_target])
    position_vector, normal_vector = parametric_line_equations(
        centroid_point_coordinates, target[2])
    target_centroid_distance = distance_between_two_points(
        centroid_point_coordinates, target[2])
    # neighbours of a neighbours
    distances_mx_i = []
    distances_mx_i_normalized = []
    for n, i in enumerate(neighbour_point_target):
        neighbour_point_neighbour = neightbours_search(
            data, i, args.n_neighbours, cosine=True)
        centroid_point_coordinates = centroid_point(
            [x[2] for x in neighbour_point_neighbour])
        position_vector, normal_vector = parametric_line_equations(
            centroid_point_coordinates, i[2])
        i_centroid_distance = distance_between_two_points(
            centroid_point_coordinates, i[2])
        distances_mx_i.append(i_centroid_distance)
        distances_mx_i_normalized.append(
            i_centroid_distance / target_centroid_distance)
    return target_centroid_distance, distances_mx_i, distances_mx_i_normalized


def main():
    args = get_args()
    model = torch.load(
    	args.autoencoder_weights_path, 
    	map_location=torch.device('cpu')
    )
    data = construct_dataset(
        args.data_path_generated, args.data_path_real, model
    )
    distances_normalized_fake = np.zeros((
        args.number_of_samples, args.n_neighbours
    ))
    distances_normalized_real = np.zeros((
        args.number_of_samples, args.n_neighbours
    ))
    for i in tqdm(range(args.number_of_samples)):
        _, orig_f, res_f = distance_comparison_to_distribution(
            data, args
        )
        _, orig_r, res_r = distance_comparison_to_distribution(
            data, args, False
        )
        distances_normalized_fake[i] = res_f
        distances_normalized_real[i] = res_r
    print('Done!')

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    plt.figure(figsize=(32, 4))
    plt.ylim(0, 1)
    plt.xlim(0, args.n_neighbours + 1)
    plt.bar(
        list(range(1, args.n_neighbours + 1)),
        distances_normalized_real.mean(axis=0),
        label='Real image distribution'
    )
    plt.bar(
        list(range(1, args.n_neighbours + 1)),
        distances_normalized_fake.mean(axis=0),
        label='Adv. image distribution'
    )
    plt.legend(loc=1, prop={'size': 20})
    plt.xticks(list(range(args.n_neighbours + 1)))
    plt.xlabel(f'Neighbours - KNN centroid distances ({args.dataset})', fontsize=26)
    plt.ylabel('Scaled distances', fontsize=26)
    plt.savefig(f'{args.dataset}_distribution.png')
    plt.show()


if __name__ == '__main__':
    main()
