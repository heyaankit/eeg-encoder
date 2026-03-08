"""
Tests for BCI IV-2a Data Loader

TDD Approach: Tests first, then implementation
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.bcic_iv_2a import BCICIV2aDataset, load_single_subject


class TestBCICIV2aDataset:
    """Test cases for BCI IV-2a dataset loader"""
    
    def test_dataset_creation(self):
        """Test dataset can be created"""
        data_dir = 'src/data/BCICIV_2a_gdf'
        if not os.path.exists(data_dir):
            pytest.skip("Dataset not found")
        
        dataset = BCICIV2aDataset(data_dir, 'A01')
        assert dataset is not None
        assert dataset.subject == 'A01'
    
    def test_load_subject_returns_correct_shape(self):
        """Test loaded data has correct shape"""
        data_dir = 'src/data/BCICIV_2a_gdf'
        if not os.path.exists(data_dir):
            pytest.skip("Dataset not found")
        
        X, y = load_single_subject(data_dir, 'A01')
        
        # Should have 288 trials (144 train + some eval)
        assert len(X.shape) == 3, "X should be 3D (trials, channels, times)"
        assert X.shape[1] == 25, f"Expected 25 channels, got {X.shape[1]}"
        assert X.shape[2] > 1000, f"Expected >1000 time points, got {X.shape[2]}"
    
    def test_labels_are_valid(self):
        """Test labels are in valid range (0-3)"""
        data_dir = 'src/data/BCICIV_2a_gdf'
        if not os.path.exists(data_dir):
            pytest.skip("Dataset not found")
        
        X, y = load_single_subject(data_dir, 'A01')
        
        assert y.min() >= 0, "Labels should be >= 0"
        assert y.max() <= 3, "Labels should be <= 3"
        assert len(np.unique(y)) <= 4, "Should have at most 4 classes"
    
    def test_class_distribution(self):
        """Test classes are reasonably balanced"""
        data_dir = 'src/data/BCICIV_2a_gdf'
        if not os.path.exists(data_dir):
            pytest.skip("Dataset not found")
        
        X, y = load_single_subject(data_dir, 'A01')
        
        unique, counts = np.unique(y, return_counts=True)
        
        # Each class should have at least 20 trials
        for count in counts:
            assert count >= 20, f"Class with only {count} trials"
    
    def test_no_nan_values(self):
        """Test data contains no NaN values"""
        data_dir = 'src/data/BCICIV_2a_gdf'
        if not os.path.exists(data_dir):
            pytest.skip("Dataset not found")
        
        X, y = load_single_subject(data_dir, 'A01')
        
        assert not np.isnan(X).any(), "Data contains NaN values"
        assert not np.isnan(y).any(), "Labels contain NaN values"
    
    def test_multiple_subjects(self):
        """Test loading multiple subjects"""
        data_dir = 'src/data/BCICIV_2a_gdf'
        if not os.path.exists(data_dir):
            pytest.skip("Dataset not found")
        
        for subject in ['A01', 'A02']:
            X, y = load_single_subject(data_dir, subject)
            assert X.shape[0] > 0, f"No data loaded for {subject}"
            assert y.shape[0] > 0, f"No labels loaded for {subject}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
