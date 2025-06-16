import numpy as np
from typing import List, Tuple

def load_diabetes() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the diabetes dataset from a tab-separated file.

    The function reads a .tab file where the first line contains the feature labels,
    and the subsequent lines contain the data values. The last value in each line is the target.

    Returns
    -------
    data : np.ndarray
        The feature data array with shape (n_samples, n_features).
    target : np.ndarray
        The target values array with shape (n_samples,).
    feature_labels : List[str]
        The list of feature labels.
    """
    with open('diabetes.tab', 'r') as f:
        lines = f.readlines()
        feature_labels = lines[0].strip().split('\t')
    
        data = []
        for line in lines[1:]:
            values = [float(n) for n in line.strip().split('\t')]
            data.append(values)
    
    data = np.array(data)
    target = data[:, -1]
    data = data[:, :-1]

    return data, target, feature_labels

def pearson_correlation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calculate the Pearson correlation coefficient between two datasets.

    Parameters
    ----------
    X : np.ndarray
        A 2D array where rows represent samples and columns represent variables.
    Y : np.ndarray
        A 2D array where rows represent samples and columns represent variables.

    Returns
    -------
    np.ndarray
        A 2D array of Pearson correlation coefficients.

    Raises
    ------
    ValueError
        If either X or Y has less than 2 samples (rows).
    """

    if X.shape[0] < 2 or Y.shape[0] < 2:
        raise ValueError("X and Y must have at least 2 samples (rows).")

    def adjust_array(x: np.ndarray):
        centered_x = x - x.mean(axis=0)
        return centered_x / np.sum(centered_x**2, axis=0)**0.5

    return adjust_array(X).T @ adjust_array(Y)


def print_table(row_names: list[str], col_names: list[str], data: list[np.ndarray], sep: str='   ') -> None:
    """
    Print a table with row and column labels, formatting numerical values to three decimal points.

    Parameters
    ----------
    row_names : list of str
        A list of strings that label each row.
    col_names : list of str
        A list of strings that label each column.
    data : list of numpy.ndarray
        A list of numpy arrays, the same length as `col_names` where each numpy array has the same length as `row_names`.
    sep : str, optional
        A string that separates each column. Default is three spaces.

    Returns
    -------
    None
    """
    
    # Ensure data is in the right format (list of numpy arrays)
    if not all(isinstance(d, np.ndarray) for d in data):
        raise ValueError("All data entries must be numpy arrays")
    
    # Convert data to list of list of strings
    data = [[f"{n:.3f}" for n in arr] for arr in zip(*data)]

    # Synthesize row_names, col_names, and data into a single list of lists
    table = [[rn] + dl for rn, dl in zip(row_names, data)]
    table = [[""] + col_names] + table

    # Find the maximum length of each column
    max_col_len = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
  
    # Pad each string in the table
    padded_table = [
        [row[i].ljust(max_col_len[i]) for i in range(len(row))]
        for row in table
    ]

    # Print the table
    for row in padded_table:
        print(*row, sep=sep)


def print_stats_table(data: np.ndarray, feature_labels: list[str]) -> None:
    """
    Print a statistics table with mean, standard deviation, minimum, maximum, and range for each feature in the data.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array containing the data to be summarized.
    feature_labels : list of str
        A list of strings that label each feature in the data.

    Returns
    -------
    None
    """
    print_table(feature_labels,
                ["Mean", "Std", "Min", "Max", "Range"],
                [data.mean(axis=0), data.std(axis=0),
                 data.min(axis=0), data.max(axis=0),
                 data.max(axis=0) - data.min(axis=0)])


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate the mean squared error between the actual and predicted values.

    Parameters
    ----------
    y : np.ndarray
        A 1D array of actual target values.
    y_hat : np.ndarray
        A 1D array of predicted target values.

    Returns
    -------
    float
        The mean squared error.

    Raises
    ------
    ValueError
        If y and y_hat do not have the same shape.
    """
    if y.shape != y_hat.shape:
        raise ValueError("y and y_hat must have the same shape.")

    return np.sum((y_hat - y)**2) / y.shape[0]


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Standardize the data to have a mean of 0 and a standard deviation of 1.

    Parameters
    ----------
    data : np.ndarray
        The input data array with shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        The standardized data array with a mean of 0 and a standard deviation of 1.
    """
    return (data - data.mean(axis=0)) / data.std(axis=0)

