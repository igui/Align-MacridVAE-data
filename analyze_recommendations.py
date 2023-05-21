import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

def get_top_k_items(x: npt.NDArray, k: int) -> npt.NDArray:
    # Best indexes without sorting
    best_indices = np.argpartition(x, axis=1, kth=-k)[:, -k:]

    # Best scores sorted in ascending order
    best_values = np.take_along_axis(x, best_indices, axis=1)

    # Best indices in descending order (from best_best_values)
    best_values_idxs = np.argsort(best_values, axis=1)[:,-1:-k-1:-1]

    # Now, we have the best indices in descending order
    return np.take_along_axis(best_indices, best_values_idxs, axis=1)

def ndcg_at_k(scores: npt.NDArray, test: sparse.csr_matrix, k: int) -> float:
    assert scores.shape == test.shape
    best_scores = get_top_k_items(scores, k)

    test = test > 0
    num_positives = np.minimum(k, test.sum(axis=1)).astype(np.int32)

    rows = np.indices(best_scores.shape)[0]
    denominator = np.log2(np.arange(2, k + 2))

    idcg = np.cumsum(1 / denominator)[np.maximum(num_positives - 1, 0)]

    dcg = np.sum(test[rows, best_scores] / denominator, axis=1)

    ndcg = dcg / idcg
    return ndcg

def recall_at_k(scores: npt.NDArray, test: sparse.csr_matrix, k: int) -> float:
    assert scores.shape == test.shape
    best_scores = get_top_k_items(scores, k)

    test = test > 0
    num_positives = np.minimum(k, test.sum(axis=1)).astype(np.int32)

    rows = np.indices(best_scores.shape)[0]
    recall = np.sum(test[rows, best_scores], axis=1) / num_positives
    return recall

def precision_at_k(scores: npt.NDArray, test: sparse.csr_matrix, k: int) -> float:
    assert scores.shape == test.shape
    best_scores = get_top_k_items(scores, k)

    test = test > 0

    rows = np.indices(best_scores.shape)[0]
    precision = np.sum(test[rows, best_scores], axis=1) / k
    return precision

def average_precision(scores: npt.NDArray, test: sparse.csr_matrix) -> float:
    assert scores.shape == test.shape

    test = test > 0
    num_positives = np.asarray(test.sum(axis=1).astype(np.int32)).reshape(-1)

    k = num_positives.max()
    best_scores = get_top_k_items(scores, k)

    rows = np.indices(best_scores.shape)[0]
    labels = test[rows, best_scores].toarray()

    true_positives_at_k = np.cumsum(labels, axis=1)
    precisions_at_k  = true_positives_at_k / np.arange(1, k+1)
    return (precisions_at_k * labels).sum(axis=1) / num_positives


def load_interaction_matrix(file: Path, shape: Tuple[int, int]) -> sparse.csr_matrix:
    df = pd.read_csv(file)
    return sparse.csr_matrix(
        (np.ones_like(df['user']), (df['user'], df['item'])),
        shape=shape,
        dtype=float
    )

def main(results_path: Path, data_path: Path):
    assert results_path.exists(), f'File {results_path} does not exist'
    assert data_path.exists(), f'Base path {data_path} does not exist'
    train_path = data_path / 'train.txt'
    assert train_path.exists(), f'File {train_path} does not exist'
    test_path = data_path / 'test.txt'
    assert test_path.exists(), f'File {test_path} does not exist'

    print(f"Loading results from {results_path}...")
    results: NDArray[np.floating] = np.vstack(next(iter(np.load(results_path, allow_pickle=True).values())))

    print(f"Results shape: {results.shape}")
    print("Loading interaction matrices...")
    train = load_interaction_matrix(data_path / 'train.txt', results.shape)
    test = load_interaction_matrix(data_path / 'test.txt', results.shape)

    print("Computing metrics...")
    results[train.nonzero()] = -np.inf

    print(f'NDCG@5:      {ndcg_at_k(results, test, 5).mean():.6f}')
    print(f'Recall@5:    {recall_at_k(results, test, 5).mean():.6f}')
    print(f'MAP:         {average_precision(results, test).mean():.6f}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", "-r", type=Path)
    parser.add_argument("--data-path", "-d", type=Path)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.results_path, args.data_path)
