import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def cluster(img, amount_clusters, colours):

    # Getting image dimensions
    y, x, depth = img.shape

    # Generate random clusters within image
    clusters = np.random.rand(amount_clusters, depth + 2) * 10

    # Create cluster mapping
    cluster_mapping = np.zeros((y, x))

    # Image used for clustering
    img_cluster = transform_img(np.copy(img), x, y)

    error_threshold = 0.001
    error = error_threshold + 1
    error_new = error_threshold

    # Iteration count
    i = 0

    while abs(error_new - error) > error_threshold and not i > 20:

        i = i + 1

        error = error_new

        # Assign clusters
        cluster_mapping = assign_clusters(img_cluster, clusters, cluster_mapping, x, y)

        # Get new clusters
        clusters, error_new = update_clusters(img_cluster, clusters, cluster_mapping, x, y)

        print(error_new)

    plot_cluster(img, cluster_mapping.astype(int), x, y, colours)


def transform_img(img, x_size, y_size):

    img.setflags(write=True)

    dummy = np.zeros((y_size, x_size, 1))

    img = np.append(img, dummy, axis=2)
    img = np.append(img, dummy, axis=2)

    for y in range(0, y_size):
        for x in range(0, x_size):

            img[y][x][3] = (y / y_size) * 50
            img[y][x][4] = (x / x_size) * 50

            img[y][x][0] = img[y][x][0] * 10 / 255.0
            img[y][x][1] = img[y][x][1] * 10 / 255.0
            img[y][x][2] = img[y][x][2] * 10 / 255.0

            #print(img[y][x])

            #space = img[y][x]
            #space = (space - min(space)) / (max(space) - min(space)) *

            #img[y][x] = space

    img.setflags(write=False)

    return img


def plot_cluster(img, cluster_mapping, x_size, y_size, colours):

    # Create segmented image and set colors
    img_segmented = np.copy(img)
    img_segmented.setflags(write=True)

    for y in range(0, y_size):
        for x in range(0, x_size):
            img_segmented[y][x] = colours[cluster_mapping[y][x]]

    # Plotting
    plt.subplot(1, 2, 1)
    plt.imshow(img, zorder=1)
    plt.subplot(1, 2, 2)
    plt.imshow(img_segmented, alpha=1, zorder=2)

    plt.show()


def assign_clusters(img, clusters, cluster_mapping, x_size, y_size):

    for y in range (0, y_size):
        for x in range (0, x_size):

            # Getting RGB vector of pixel
            pixel = img[y][x]

            # Find closest RGB colour-space cluster
            cluster_index = closest_cluster(pixel, clusters)

            # Update cluster mapping accordingly
            cluster_mapping[y, x] = cluster_index

    return cluster_mapping


def closest_cluster(pixel, clusters):

    return np.linalg.norm(clusters - pixel, axis=1).argmin()


def update_clusters(img, clusters, cluster_mapping, x_size, y_size):

    old_clusters = np.copy(clusters)

    for index, _ in enumerate(clusters):

        # Find all pixels belonging to this cluster
        pixels = [img[y][x] for y in range (0, y_size) for x in range (0, x_size) if cluster_mapping[y][x] == index]

        if len(pixels) > 0:
            clusters[index] = np.mean(pixels)

    error = np.linalg.norm(clusters - old_clusters)

    return clusters, error


def get_rgb(colours):

    colours_rgb = []

    colour_dict = {
        'r': [255, 0 ,0],
        'g': [0, 255, 0],
        'b': [0, 0, 255],
        'y': [255, 255, 0],
        'c': [0, 255, 255],
        'm': [255, 0, 255]
    }

    for colour in colours:
        colours_rgb.append(colour_dict[colour])

    return colours_rgb


def main():

    clusters = 3

    colours = ['r', 'g', 'b', 'y', 'c']
    colours_rgb = get_rgb(colours)

    img = mpimg.imread('img_1.jpg')

    cluster(img, clusters, colours_rgb)


if __name__ == '__main__':
    main()

