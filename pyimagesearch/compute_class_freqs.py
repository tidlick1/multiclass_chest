import numpy as np

def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """

    # total number of patients (rows)
    numrow, num_cols = labels.shape
    N = numrow
    positive_frequencies = (np.count_nonzero(labels, axis=0)) / N
    negative_frequencies = (N - (np.count_nonzero(labels, axis=0))) / N
    return positive_frequencies, negative_frequencies