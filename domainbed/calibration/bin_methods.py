# coding=utf-8
# Copyright 2021 Hiroki Naganuma (Hiroki11x).

# This code is based on following repository
# https://github.com/google-research/google-research/tree/master/caltrain


"""Binning methods."""
import abc
import numpy as np


class BinMethod(abc.ABC):
    """General interface for specifying binning method."""

    def __init__(self, num_bins):
        self.num_bins = num_bins

    @abc.abstractmethod
    def compute_bin_indices(self, scores):
        """Assign a bin index for each score.
        Args:
          scores: np.array of shape (num_examples, num_classes) containing the
            model's confidence scores
        Returns:
          bin_indices: np.array of shape (num_examples, num_classes) containing the
            bin assignment for each score
        """
        pass


class BinEqualWidth(BinMethod):
    """Divide the scores into equal-width bins."""

    def compute_bin_indices(self, scores):
        """Assign a bin index for each score assuming equal width bins.
        Args:
          scores: np.array of shape (num_examples, num_classes) containing the
            model's confidence scores
        Returns:
          bin_indices: np.array of shape (num_examples, num_classes) containing the
            bin assignment for each score
        """
        edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        bin_indices = np.digitize(scores, edges, right=False)
        # np.digitze uses one-indexed bins, switch to using 0-indexed
        bin_indices = bin_indices - 1
        # Put examples with score equal to 1.0 in the last bin.
        bin_indices = np.where(scores == 1.0, self.num_bins - 1, bin_indices)
        return bin_indices


class BinEqualExamples(BinMethod):
    """Divide the scores into bins with equal number of examples."""

    def compute_bin_indices(self, scores):
        """Assign a bin index for each score assumes equal num examples per bin.
        Args:
          scores: np.ndarray of shape [N, K] containing the model's confidence
        Returns:
          bin_indices: np.ndarray of shape [N, K] containing the bin assignment for
            each score
        """
        num_examples = scores.shape[0]
        num_classes = scores.shape[1]

        bin_indices = np.zeros((num_examples, num_classes), dtype=int)
        for k in range(num_classes):
            sort_ix = np.argsort(scores[:, k])
            bin_indices[:, k][sort_ix] = np.minimum(
                self.num_bins - 1,
                np.floor((np.arange(num_examples) / num_examples) *
                         self.num_bins)).astype(int)
        return bin_indices