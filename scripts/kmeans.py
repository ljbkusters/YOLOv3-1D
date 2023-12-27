"""A module to perform kmeans clustering to determine anchor boxes

Performs kmeans-clustering on 1D datasets
"""

import os
import model.utils
import numpy
import tqdm


def euclidean_1d(width, mean):
    distance = abs(mean - width)
    return distance


def iou_distance(box, mean_box):
    pass


def load_bounding_domains(label_dir_path: os.PathLike):
    labels = []
    for filename in tqdm.tqdm(os.listdir(label_dir_path)):
        if filename.endswith(".csv"):
            file_path = os.path.join(label_dir_path, filename)
            labels.extend(numpy.genfromtxt(fname=file_path,
                                           delimiter=" ",
                                           comments="#",
                                           ndmin=2).tolist())
    labels = numpy.array(labels)
    bounding_domains = labels[:, 0:2]  # throw away label classes, it is not required
    return bounding_domains.T


def k_means_clustering(domains: numpy.ndarray,
                       k: int,
                       n_trials: int=10,
                       max_iterations: int=300):
    num_domains = domains.shape[-1]
    widths = domains[1]
    cluster_means_and_variances = \
        numpy.zeros((n_trials, k, 2), dtype=float)
    for trial_idx in range(n_trials):
        # choose random intialization
        last_cluster_means = numpy.random.choice(widths, size=(k,), replace=False)
        variances = None
        for iter_idx in range(max_iterations):
            # determine distances from means
            distances = numpy.zeros((k, num_domains), dtype=float)
            for cluster_idx in range(k):
                distances[cluster_idx, :] = abs(widths - last_cluster_means[cluster_idx])
            # calculate cluster index of minimum distances
            cluster_indices = numpy.argmin(distances, axis=0)
            # assign clusters
            clusters = [[] for _ in range(k)]
            for domain_idx, cluster_idx in enumerate(cluster_indices):
                clusters[cluster_idx].append(widths[domain_idx])
            # recalculate means for each cluster
            new_cluster_means = numpy.array([numpy.mean(cluster) for cluster in clusters])
            variances = numpy.array([numpy.var(cluster) for cluster in clusters])
            # convergence determined when means of this iter same as means of last iter
            if numpy.isclose(new_cluster_means, last_cluster_means, rtol=1e-4).all():
                break
            elif (iter_idx + 1) == max_iterations:
                raise UserWarning("KMeans clustering did not converge on"
                                  f" trial {trial_idx} after {max_iterations}"
                                  " iterations.")
            else:
                last_cluster_means = new_cluster_means
        cluster_means_and_variances[trial_idx, :, 0] = new_cluster_means
        cluster_means_and_variances[trial_idx, :, 1] = variances
    # the best clusters are the ones with the minimum sum of variances
    min_varsum_cluster_means_idx = numpy.argmin(numpy.sum(
        cluster_means_and_variances[:, :, 1], axis=1))
    # finally, return the optimal cluster means and variances
    return cluster_means_and_variances[min_varsum_cluster_means_idx, :, :]

def elbow_plot(domains):
    print("calculating kmeans...")
    variances = []
    for k in tqdm.tqdm(range(1, 9+1)):
        means_and_vars = k_means_clustering(domains, k, n_trials=10, max_iterations=300)
        sumvar = numpy.sum(means_and_vars[:, 1])
        variances.append(sumvar)

    import matplotlib.pyplot as plt
    plt.plot(range(2, 9+1), numpy.diff(variances))
    plt.show()

def calculate_optimal_domain_anchors(domains, k=6):
    means_and_vars = k_means_clustering(domains, k, n_trials=10, max_iterations=300)
    means = means_and_vars[:, 0]
    print(numpy.sort(means))

def main():
    print("loading domains...")
    domains = load_bounding_domains(os.path.join("data", "test_data", "labels"))
    #elbow_plot(domains)
    calculate_optimal_domain_anchors(domains)
if __name__ == "__main__":
    main()
