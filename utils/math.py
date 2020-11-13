import numpy as np


def centroid_point(data_points):
    sum_ = np.sum(data_points, axis=0)
    return sum_ / len(data_points)


def parametric_line_equations(start_point, end_point):
    '''
    for reference http://sites.science.oregonstate.edu/math/home/programs/undergrad/CalculusQuestStudyGuides/vcalc/lineplane/lineplane.html
    takes two points as input and outputs parametric representation of n-dim line
    x = x_1 + (x_2 - x_1) * t
    y = y_1 + (y_2 - y_1) * t
    z = z_1 + (z_2 - z_1) * t
    ...
    output:
    [x, y, z, ...] = start_point + t * (end_point - start_point)

    '''
    position_vector = start_point
    direction_vector = end_point - start_point
    return position_vector, direction_vector


def distance_between_two_points(point_a, point_b):
    '''
    Euclidean distance between two points
    '''
    distance = np.linalg.norm(point_a - point_b)
    return distance


def perpendicular_plane_equation(normal_vector, position_vector):
    '''
    Hyperplane equation : Ax + By + Cz + ... = bias
    A, B, C, ... are equal to the corresponding elements of normal_vector
    bias is calculated by: A * x_0 + B * y_0 + ... = bias where x_0, y_0, z_0 ... are elemts of position_vetor
    '''
    weights = normal_vector
    bias = np.matmul(normal_vector, position_vector)
    return weights, bias


def distance_from_point_to_hyperplane(point, plane_weights, plane_bias):
    numerator = abs(np.dot(point, plane_weights) - plane_bias)
    denominator = np.linalg.norm(plane_weights)
    return numerator / denominator


def same_side_as_target_point(point, target, plane_weights, plane_bias):
    target_sign = np.dot(plane_weights, target) - plane_bias
    point_sign = np.dot(plane_weights, point) - plane_bias
    return np.sign(target_sign) == np.sign(point_sign)


def cosine_similarity(image_a, image_b):
    dot = np.dot(image_a, image_b)
    norma = np.linalg.norm(image_a)
    normb = np.linalg.norm(image_b)
    cos = dot / (norma * normb)
    return cos


def euclidean_distance(image_a, image_b):
    distance = np.linalg.norm(image_a - image_b)
    return distance
