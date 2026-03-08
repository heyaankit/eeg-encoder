"""
ZUNA Preprocessing Pipeline for EEG Motor Imagery

This module handles the ZUNA EEG foundation model integration for denoising
and reconstructing EEG signals before classification.

ZUNA Workflow:
1. Convert GDF to FIF format
2. Set channel montage
3. Run ZUNA preprocessing (filter, epoch, normalize)
4. Run ZUNA inference (denoise)
5. Use denoised data for training

Reference: https://github.com/Zyphra/zuna
"""

import os
import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Optional, List
import torch


class ZUNAPreprocessor:
    """ZUNA preprocessing pipeline for EEG data."""
    
    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        use_zuna: bool = True
    ):
        """
        Initialize ZUNA preprocessor.
        
        Args:
            data_dir: Path to raw EEG data (GDF files)
            output_dir: Path for processed data
            use_zuna: Whether to use ZUNA (if False, use basic preprocessing)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_zuna = use_zuna
        
        # ZUNA expects 256 Hz, BCI IV-2a is 250 Hz
        self.target_sfreq = 256
        self.original_sfreq = 250
        
    def gdf_to_fif(self, gdf_path: str, subject: str) -> str:
        """
        Convert GDF file to FIF format for ZUNA.
        
        Args:
            gdf_path: Path to GDF file
            subject: Subject ID
            
        Returns:
            Path to converted FIF file
        """
        print(f"Converting {gdf_path} to FIF format...")
        
        # Load GDF file
        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
        
        # Set standard 10-20 montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn')
        
        # Resample to 256 Hz (ZUNA requirement)
        raw.resample(self.target_sfreq, verbose=False)
        
        # Save as FIF
        output_path = self.output_dir / f"{subject}_raw.fif"
        raw.save(output_path, overwrite=True, verbose=False)
        
        print(f"Saved to {output_path}")
        return str(output_path)
    
    def preprocess_with_zuna(
        self,
        fif_path: str,
        subject: str,
        gpu_device: int = 0
    ) -> Tuple[np.ndarray, dict]:
        """
        Run ZUNA preprocessing and inference.
        
        Args:
            fif_path: Path to FIF file
            subject: Subject ID
            gpu_device: GPU device ID
            
        Returns:
            Denoised EEG data and metadata
        """
        try:
            import zuna
        except ImportError:
            print("ZUNA not installed. Using basic preprocessing instead.")
            return self.preprocess_basic(fif_path, subject)
        
        print(f"Running ZUNA preprocessing for {subject}...")
        
        # Step 1: ZUNA preprocessing (FIF -> PT)
        pt_input_dir = self.output_dir / "1_pt_input"
        pt_input_dir.mkdir(parents=True, exist_ok=True)
        
        zuna.preprocessing(
            input_dir=str(self.output_dir),
            output_dir=str(pt_input_dir),
            apply_notch_filter=False,
            apply_highpass_filter=True,
            apply_average_reference=True,
            preprocessed_fif_dir=str(self.output_dir / "1_fif_filter"),
            verbose=False
        )
        
        # Step 2: ZUNA inference (denoising)
        pt_output_dir = self.output_dir / "2_pt_output"
        pt_output_dir.mkdir(parents=True, exist_ok=True)
        
        zuna.inference(
            input_dir=str(pt_input_dir),
            output_dir=str(pt_output_dir),
            gpu_device=str(gpu_device),
            diffusion_cfg=1.0,
            diffusion_sample_steps=50,
            verbose=False
        )
        
        # Step 3: Convert back to FIF
        fif_output_dir = self.output_dir / "3_fif_output"
        fif_output_dir.mkdir(parents=True, exist_ok=True)
        
        zuna.pt_to_fif(
            input_dir=str(pt_output_dir),
            output_dir=str(fif_output_dir),
            verbose=False
        )
        
        # Load the denoised data
        denoised_raw = mne.io.read_raw_fif(
            str(list(fif_output_dir.glob("*.fif"))[0]),
            verbose=False
        )
        
        data = denoised_raw.get_data()
        info = denoised_raw.info
        
        metadata = {
            'subject': subject,
            'sfreq': info['sfreq'],
            'channels': len(info['ch_names']),
            'n_times': data.shape[1]
        }
        
        return data, metadata
    
    def preprocess_basic(
        self,
        fif_path: str,
        subject: str
    ) -> Tuple[np.ndarray, dict]:
        """
        Basic preprocessing without ZUNA (fallback).
        
        Args:
            fif_path: Path to FIF file
            subject: Subject ID
            
        Returns:
            Preprocessed EEG data and metadata
        """
        print(f"Running basic preprocessing for {subject}...")
        
        # Load data
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        
        # Bandpass filter (0.5-100 Hz)
        raw.filter(l_freq=0.5, h_freq=100, verbose=False)
        
        # Notch filter (50 Hz)
        raw.notch_filter(freqs=50, verbose=False)
        
        # Get data
        data = raw.get_data()
        
        metadata = {
            'subject': subject,
            'sfreq': raw.info['sfreq'],
            'channels': len(raw.ch_names),
            'n_times': data.shape[1],
            'method': 'basic'
        }
        
        return data, metadata
    
    def process_subject(
        self,
        subject: str,
        gpu_device: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single subject's data through ZUNA.
        
        Args:
            subject: Subject ID (e.g., 'A01')
            gpu_device: GPU device for ZUNA
            
        Returns:
            X: EEG data (n_trials, n_channels, n_times)
            y: Labels
        """
        # Find GDF files
        train_file = self.data_dir / f"{subject}T.gdf"
        
        if not train_file.exists():
            raise FileNotFoundError(f"No data found for {subject}")
        
        # Convert to FIF
        fif_path = self.gdf_to_fif(str(train_file), subject)
        
        # Run preprocessing
        if self.use_zuna:
            data, metadata = self.preprocess_with_zuna(fif_path, subject, gpu_device)
        else:
            data, metadata = self.preprocess_basic(fif_path, subject)
        
        print(f"Processed {subject}: {metadata}")
        
        return data, metadata
    
    def process_all_subjects(
        self,
        subjects: Optional[List[str]] = None,
        gpu_device: int = 0
    ) -> dict:
        """
        Process all subjects.
        
        Args:
            subjects: List of subject IDs
            gpu_device: GPU device for ZUNA
            
        Returns:
            Dictionary of subject -> (data, metadata)
        """
        if subjects is None:
            subjects = [f"A{str(i).zfill(2)}" for i in range(1, 10)]
        
        results = {}
        
        for subject in subjects:
            print(f"\n{'='*50}")
            print(f"Processing subject: {subject}")
            print('='*50)
            
            try:
                data, metadata = self.process_subject(subject, gpu_device)
                results[subject] = (data, metadata)
                print(f"Processed {subject}: shape={data.shape}")
            except Exception as e:
                print(f"Error processing {subject}: {e}")
        
        return results


def preprocess_single_subject(
    gdf_path: str,
    output_dir: str,
    use_zuna: bool = True,
    gpu_device: int = 0
) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to preprocess a single subject.
    
    Args:
        gdf_path: Path to GDF file
        output_dir: Output directory
        use_zuna: Use ZUNA or basic preprocessing
        gpu_device: GPU device
        
    Returns:
        Preprocessed data and metadata
    """
    preprocessor = ZUNAPreprocessor(
        data_dir=os.path.dirname(gdf_path),
        output_dir=output_dir,
        use_zuna=use_zuna
    )
    
    subject = os.path.basename(gdf_path).replace('.gdf', '').replace('T', '').replace('E', '')
    
    fif_path = preprocessor.gdf_to_fif(gdf_path, subject)
    
    if use_zuna:
        return preprocessor.preprocess_with_zuna(fif_path, subject, gpu_device)
    else:
        return preprocessor.preprocess_basic(fif_path, subject)


# Test function
if __name__ == '__main__':
    # Test basic preprocessing (without ZUNA for quick testing)
    data_dir = 'src/data/BCICIV_2a_gdf'
    output_dir = 'src/data/processed'
    
    print("Testing ZUNA Preprocessing Pipeline")
    print("="*50)
    
    # Use basic preprocessing (faster for testing)
    preprocessor = ZUNAPreprocessor(
        data_dir=data_dir,
        output_dir=output_dir,
        use_zuna=False  # Use basic for now
    )
    
    # Test with subject A01
    try:
        data, metadata = preprocessor.process_subject('A01')
        print(f"\n✓ Successfully processed A01")
        print(f"  Data shape: {data.shape}")
        print(f"  Metadata: {metadata}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
