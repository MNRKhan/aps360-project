import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def cluster(img, amount_clusters, colours):

    # Getting image dimensions
    y, x, depth = img.shape

    # Generate random clusters within image
    clusters = np.random.rand(amount_clusters, depth) * 255

    # Create cluster mapping
    cluster_mapping = np.zeros((y, x))

    error_threshold = 5
    error = error_threshold + 1

    while error > error_threshold:

        # Assign clusters
        cluster_mapping = assign_clusters(img, clusters, cluster_mapping, x, y)

        # Get new clusters
        clusters, error = update_clusters(img, clusters, cluster_mapping, x, y)

    plot_cluster(img, cluster_mapping.astype(int), x, y, colours)


def plot_cluster(img, cluster_mapping, x_size, y_size, colours):

    # Create segmented image and set colors
    img_segmented = np.copy(img)
    img_segmented.setflags(write=1)

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

    for index, cluster in enumerate(clusters):

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

