"""
Tests for EEGEncoder Model

TDD Approach: Tests first, then implementation
"""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.eegencoder import EEGEncoder, create_eegencoder


class TestEEGEncoder:
    """Test cases for EEGEncoder model"""
    
    def test_model_creation(self):
        """Test model can be created with default parameters"""
        model = create_eegencoder()
        assert model is not None
        assert isinstance(model, EEGEncoder)
    
    def test_model_output_shape(self):
        """Test model output has correct shape"""
        model = create_eegencoder()
        batch_size = 4
        x = torch.randn(batch_size, 1, 22, 1125)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 4), f"Expected (4, 4), got {output.shape}"
    
    def test_model_parameter_count(self):
        """Test model has reasonable parameter count"""
        model = create_eegencoder()
        param_count = sum(p.numel() for p in model.parameters())
        
        # Should be around 73K parameters
        assert 50000 < param_count < 100000, f"Parameter count {param_count} outside expected range"
    
    def test_model_forward_different_batch_sizes(self):
        """Test model works with different batch sizes"""
        model = create_eegencoder()
        
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, 1, 22, 1125)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 4)
    
    def test_model_gradient_flow(self):
        """Test gradients flow through model"""
        model = create_eegencoder()
        x = torch.randn(4, 1, 22, 1125, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None
    
    def test_model_eval_mode(self):
        """Test model works in eval mode"""
        model = create_eegencoder()
        model.eval()
        
        x = torch.randn(4, 1, 22, 1125)
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        # Outputs should be deterministic in eval mode
        assert torch.allclose(output1, output2)
    
    def test_model_input_range(self):
        """Test model handles different input ranges"""
        model = create_eegencoder()
        
        # Test with normalized input
        x_normalized = torch.randn(4, 1, 22, 1125)
        with torch.no_grad():
            output = model(x_normalized)
        assert not torch.isnan(output).any()
        
        # Test with large input
        x_large = torch.randn(4, 1, 22, 1125) * 10
        with torch.no_grad():
            output = model(x_large)
        assert not torch.isnan(output).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
