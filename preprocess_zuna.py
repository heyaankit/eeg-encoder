"""
ZUNA Preprocessing Script for BCI IV-2a Dataset

This script:
1. Runs ZUNA denoising on all subjects
2. Caches the denoised data to disk
3. Then you can train without ZUNA processing overhead

ZUNA runs on CPU to keep GPU free for training.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mne
from pathlib import Path
import json
from tqdm import tqdm


def preprocess_subject_with_zuna(subject: str, data_dir: str, output_dir: str):
    """Preprocess a single subject with ZUNA."""
    from src.preprocessing.motor_imagery_pipeline import MotorImageryPreprocessor

    print(f"\n{'=' * 50}")
    print(f"Processing subject: {subject}")
    print("=" * 50)

    # Check if cached data exists
    cached_file = Path(output_dir) / f"{subject}_denoised.npz"
    if cached_file.exists():
        print(f"  Using cached data from {cached_file}")
        return True

    # Initialize preprocessor with ZUNA
    preprocessor = MotorImageryPreprocessor(
        data_dir=data_dir,
        output_dir=output_dir,
        use_zuna=True,  # Enable ZUNA
        filter_alpha_beta=True,
    )

    try:
        # Load and preprocess with ZUNA
        X, y, metadata = preprocessor.load_and_preprocess(subject)

        # Save to cache
        np.savez_compressed(cached_file, X=X, y=y, metadata=json.dumps(metadata))
        print(f"  Saved denoised data to {cached_file}")
        print(f"  Shape: {X.shape}")

        return True

    except Exception as e:
        print(f"  Error processing {subject}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    data_dir = "src/data/BCICIV_2a_gdf"
    output_dir = "src/data/BCICIV_2a_gdf/processed/zuna_cache"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    subjects = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"]

    print("=" * 60)
    print("ZUNA Preprocessing for BCI IV-2a Dataset")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Subjects: {subjects}")
    print("=" * 60)

    # Process each subject
    results = {}
    for subject in tqdm(subjects, desc="Processing subjects"):
        success = preprocess_subject_with_zuna(subject, data_dir, output_dir)
        results[subject] = success

    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete")
    print("=" * 60)
    for subject, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {subject}")

    successful = sum(results.values())
    print(f"\nTotal: {successful}/{len(subjects)} subjects processed successfully")

    if successful == len(subjects):
        print("\nAll data preprocessed! You can now train with:")
        print("  python train_complete.py --all --use-cached-zuna")
    else:
        print("\nSome subjects failed. Check errors above.")


if __name__ == "__main__":
    main()
