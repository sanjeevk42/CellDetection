from munkres import Munkres
from scipy.spatial.distance import cdist


def  get_matches(detections, annotations, max_dist=20):
    '''Matches detections and annotations, and return indices mat

    Detections and annotations are matched with the Hungarian algorithm.

    Parameters
    ----------
    detections : array_like, shape=[n_points, n_dim]
        detected cells.
    annotations : array_like, shape=[n_points, n_dim]
        annotated cells.
    max_dist : float
        maximum distance to be considered as a match.

    Returns
    -------
    indices : int x int' list

    '''
    matrix = cdist(detections, annotations, 'euclidean')
    matrix[matrix > max_dist] = 100 * max_dist

    indices = Munkres().compute(matrix.tolist())

    new_indices = []
    for idx in indices:
        if matrix[idx] <= max_dist:
            new_indices.append(idx)
    return new_indices


def score_detections(detections, annotations, indices):
    '''calculates detection accuracy in terms of precision,
    recall and F1 score.

    Detections and annotations are matched with the Hungarian algorithm.

    Parameters
    ----------
    detections : array_like, shape=[n_points, n_dim]
        detected cells.
    annotations : array_like, shape=[n_points, n_dim]
        annotated cells.
    indices : int x int' list
        indices of matches, obtained with detection.get_matches

    Returns
    -------
    recall, precision, f1 : float

    Notes
    -----precision and recall are zero, then we define
    If both
    the F1 score as 0.

    '''

    # count how many of the matches are closer than max_dist:
    # this is the number of true positives.
    tp = len(indices)
    if tp == 0:
        return 0., 0., 0.

    # derive the remaining scores.
    recall = tp / float(len(annotations))
    precision = tp / float(len(detections))

    f1 = 2 * recall * precision / (precision + recall)

    return recall, precision, f1
