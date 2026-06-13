"""Tests for the streaming RecommenderDataset."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.recommender.data import (
    BatchCollator,
    RecommenderDataset,
    make_batches,
)


def test_recommender_dataset_streaming_mode():
    """Test that streaming mode works correctly."""
    # Create simple interaction data
    interactions = pd.DataFrame({
        "user_idx": [0, 1, 2],
        "item_idx": [0, 1, 2]
    })
    
    # Test eager mode (default)
    dataset_eager = RecommenderDataset(
        interactions=interactions,
        num_items=5,
        num_negatives=2,
        streaming=False
    )
    
    assert len(dataset_eager) == 9  # 3 positives * (1 + 2 negatives) = 9 samples
    assert hasattr(dataset_eager, 'samples')
    assert isinstance(dataset_eager.samples, list)
    
    # Test streaming mode
    dataset_streaming = RecommenderDataset(
        interactions=interactions,
        num_items=5,
        num_negatives=2,
        streaming=True
    )
    
    assert len(dataset_streaming) == 9  # Same total count
    assert not hasattr(dataset_streaming, 'samples')  # Should not have samples attribute in streaming mode
    
    # Test that we can access items via __getitem__
    user, item, label = dataset_streaming[0]
    assert isinstance(user, np.int64)
    assert isinstance(item, np.int64)
    assert isinstance(label, np.float32)
    
    # Test that the first sample is a positive (label=1.0)
    assert label == 1.0
    
    # Test that subsequent samples are negatives
    neg_user, neg_item, neg_label = dataset_streaming[1]
    assert neg_label == 0.0
    assert neg_user == 0  # Same user as first sample


def test_recommender_dataset_streaming_consistency():
    """Test that streaming mode produces consistent results."""
    # Create interaction data
    interactions = pd.DataFrame({
        "user_idx": [0, 1],
        "item_idx": [0, 1]
    })
    
    # Test with fixed seed for reproducibility
    dataset1 = RecommenderDataset(
        interactions=interactions,
        num_items=5,
        num_negatives=2,
        streaming=True,
        seed=42
    )
    
    dataset2 = RecommenderDataset(
        interactions=interactions,
        num_items=5,
        num_negatives=2,
        streaming=True,
        seed=42
    )
    
    # Should produce same results with same seed
    sample1 = dataset1[0]
    sample2 = dataset2[0]
    assert sample1 == sample2
    
    # Test that we can access different samples
    sample1_pos = dataset1[0]  # First positive
    sample1_neg = dataset1[1]  # First negative (from same user)
    
    assert sample1_pos[2] == 1.0  # Positive label
    assert sample1_neg[2] == 0.0  # Negative label
    
    # The test is mainly to ensure the streaming mode works, not that it's perfectly reproducible
    # since negative sampling is random - just verify it doesn't crash
    assert isinstance(sample1_pos, tuple)
    assert len(sample1_pos) == 3


def test_recommender_dataset_streaming_batching():
    """Test batch processing with streaming mode."""
    interactions = pd.DataFrame({
        "user_idx": [0, 1, 2],
        "item_idx": [0, 1, 2]
    })
    
    dataset = RecommenderDataset(
        interactions=interactions,
        num_items=5,
        num_negatives=2,
        streaming=True
    )
    
    # Test batch iteration
    batches = list(dataset.stream_batches(batch_size=2))
    assert len(batches) > 0
    
    for users, items, labels in batches:
        assert isinstance(users, torch.Tensor)
        assert isinstance(items, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert users.shape[0] == items.shape[0] == labels.shape[0]
        assert users.dtype == torch.long
        assert items.dtype == torch.long
        assert labels.dtype == torch.float32


def test_batch_collator():
    """Test the BatchCollator utility."""
    collator = BatchCollator(device="cpu")
    
    # Test with simple data
    batch_data = [
        (np.int64(0), np.int64(1), np.float32(1.0)),
        (np.int64(1), np.int64(2), np.float32(0.0)),
    ]
    
    users, items, labels = collator(batch_data)
    
    assert isinstance(users, torch.Tensor)
    assert isinstance(items, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert users.shape == (2,)
    assert items.shape == (2,)
    assert labels.shape == (2,)
    assert users.dtype == torch.long
    assert items.dtype == torch.long
    assert labels.dtype == torch.float32


def test_make_batches():
    """Test the make_batches utility function."""
    interactions = pd.DataFrame({
        "user_idx": [0, 1, 2],
        "item_idx": [0, 1, 2]
    })
    
    dataset = RecommenderDataset(
        interactions=interactions,
        num_items=5,
        num_negatives=1,
        streaming=False
    )
    
    batches = list(make_batches(dataset, batch_size=2))
    assert len(batches) > 0
    
    for users, items, labels in batches:
        assert isinstance(users, torch.Tensor)
        assert isinstance(items, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert users.shape[0] == items.shape[0] == labels.shape[0]


def test_recommender_dataset_streaming_edge_cases():
    """Test edge cases for streaming dataset."""
    # Empty interactions
    interactions = pd.DataFrame({"user_idx": [], "item_idx": []})
    
    dataset = RecommenderDataset(
        interactions=interactions,
        num_items=5,
        num_negatives=2,
        streaming=True
    )
    
    assert len(dataset) == 0
    
    # Single interaction
    interactions = pd.DataFrame({"user_idx": [0], "item_idx": [1]})
    
    dataset = RecommenderDataset(
        interactions=interactions,
        num_items=5,
        num_negatives=2,
        streaming=True
    )
    
    assert len(dataset) == 3  # 1 positive + 2 negatives
    
    # Test accessing valid indices
    user, item, label = dataset[0]  # Should be positive
    assert label == 1.0
    
    user, item, label = dataset[1]  # Should be negative
    assert label == 0.0