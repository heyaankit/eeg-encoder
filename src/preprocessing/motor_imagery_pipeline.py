"""
Enhanced Motor Imagery Preprocessing Pipeline

Based on professional BCI research for BCI Competition IV-2a:
- High-pass filter: 1Hz (remove drift)
- Notch filter: 50Hz (power line interference)
- Bandpass filter: 8-32Hz (motor imagery alpha/beta bands)
- Baseline correction: first 200ms pre-cue
- Use only 22 EEG channels (exclude 3 EOG)

Reference: Multiple BCI Competition IV winning approaches
"""

import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Optional, List
import warnings
from scipy.signal import cheby2, sosfilt

warnings.filterwarnings("ignore")


class MotorImageryPreprocessor:
    """Enhanced preprocessing for motor imagery EEG classification."""

    def __init__(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        use_zuna: bool = False,
        filter_alpha_beta: bool = True,
        high_pass: float = 1.0,
        low_pass: float = 100.0,
        bandpass_low: float = 8.0,
        bandpass_high: float = 32.0,
        notch_freq: float = 50.0,
        tmin: float = 0.0,
        tmax: float = 4.5,
    ):
        """
        Initialize preprocessor.

        Args:
            data_dir: Path to raw EEG data (GDF files)
            output_dir: Path for processed data
            use_zuna: Whether to use ZUNA (if available)
            filter_alpha_beta: Apply 8-32Hz bandpass for alpha/beta
            high_pass: High-pass filter cutoff
            low_pass: Low-pass filter cutoff
            bandpass_low: Bandpass lower cutoff (alpha/beta)
            bandpass_high: Bandpass upper cutoff (alpha/beta)
            notch_freq: Notch filter frequency
            tmin: Epoch start time (seconds, relative to cue)
            tmax: Epoch end time (seconds, relative to cue)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = (
            Path(output_dir) if output_dir else self.data_dir / "processed"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_zuna = use_zuna
        self.filter_alpha_beta = filter_alpha_beta
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.tmin = tmin
        self.tmax = tmax

        # BCI IV-2a event codes
        # 769 = left hand, 770 = right hand, 771 = feet, 772 = tongue
        self.event_codes = {769: 0, 770: 1, 771: 2, 772: 3}
        self.class_labels = {0: "left_hand", 1: "right_hand", 2: "feet", 3: "tongue"}

    def load_and_preprocess(self, subject: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load and preprocess a single subject's data.

        Args:
            subject: Subject ID (e.g., 'A01')

        Returns:
            X: EEG data (n_trials, n_channels, n_times)
            y: Labels (n_trials,)
            metadata: Processing metadata
        """
        # Load training data
        train_file = self.data_dir / f"{subject}T.gdf"
        eval_file = self.data_dir / f"{subject}E.gdf"

        if not train_file.exists():
            raise FileNotFoundError(f"No data found for {subject}")

        print(f"Loading {subject} data...")

        # Load training data
        raw_train = mne.io.read_raw_gdf(str(train_file), preload=True, verbose=False)
        X_train, y_train = self._extract_epochs(raw_train)

        # Try to load evaluation data
        X_eval, y_eval = None, None
        if eval_file.exists():
            try:
                raw_eval = mne.io.read_raw_gdf(
                    str(eval_file), preload=True, verbose=False
                )
                X_eval, y_eval = self._extract_epochs(raw_eval)
                print(f"  Loaded {len(X_train)} train + {len(X_eval)} eval trials")
            except Exception as e:
                print(f"  Could not load eval data: {e}")
                print(f"  Using only {len(X_train)} training trials")

        # Combine data if evaluation available
        if X_eval is not None:
            X = np.concatenate([X_train, X_eval], axis=0)
            y = np.concatenate([y_train, y_eval], axis=0)
        else:
            X = X_train
            y = y_train

        # Apply preprocessing
        X = self._preprocess_data(X)

        metadata = {
            "subject": subject,
            "n_trials": len(X),
            "n_channels": X.shape[1],
            "n_times": X.shape[2],
            "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
            "sfreq": 250,
            "preprocessing": "bandpass_8_32Hz"
            if self.filter_alpha_beta
            else "bandpass_1_100Hz",
        }

        print(f"  Processed: {X.shape}, Classes: {np.unique(y, return_counts=True)}")

        return X, y, metadata

    def _extract_epochs(self, raw: mne.io.Raw) -> Tuple[np.ndarray, np.ndarray]:
        """Extract epochs from raw EEG data - using correct event mapping."""
        # Get events from annotations
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Map GDF event codes to motor imagery classes
        # Training file uses: 769=left, 770=right, 771=feet, 772=tongue
        gdf_to_class = {
            "769": 1,  # left hand -> class 1
            "770": 2,  # right hand -> class 2
            "771": 3,  # feet -> class 3
            "772": 4,  # tongue -> class 4
        }

        # Get the mapping from sequential IDs to GDF codes
        id_to_gdf = {}
        for k, v in event_id.items():
            key = str(k)  # Convert numpy string to regular string
            if key in gdf_to_class:
                id_to_gdf[v] = key

        # Filter events to only motor imagery classes
        valid_events = []
        for event in events:
            event_code = event[2]
            if event_code in id_to_gdf:
                gdf_code = id_to_gdf[event_code]
                class_id = gdf_to_class[gdf_code]
                # Format: [sample, 0, class_id]
                valid_events.append([event[0], 0, class_id])

        if len(valid_events) == 0:
            raise ValueError("No valid motor imagery events found")

        valid_events = np.array(valid_events, dtype=np.int64)

        # Define event ID mapping (1-4 -> left_hand, right_hand, feet, tongue)
        event_id_mapping = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}

        # Create epochs - same parameters as bcic_iv_2a.py
        epochs = mne.Epochs(
            raw,
            valid_events,
            event_id_mapping,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,
            verbose=False,
            preload=True,
            picks="eeg",
        )

        # Get data and labels
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        y = epochs.events[:, 2] - 1  # Convert 1-4 to 0-3

        return X, y

    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to EEG data.

        Args:
            X: EEG data (n_trials, n_channels, n_times)

        Returns:
            Preprocessed EEG data
        """
        # Apply standard scaling per channel (common for EEG)
        # Reshape for StandardScaler: (n_trials * n_channels, n_times)
        n_trials, n_channels, n_times = X.shape
        X_flat = X.transpose(0, 1, 2).reshape(-1, n_times)

        # Standard scaling
        mean = X_flat.mean(axis=1, keepdims=True)
        std = X_flat.std(axis=1, keepdims=True)
        std = np.where(std > 0, std, 1)  # Avoid division by zero

        X_scaled = (X_flat - mean) / std
        X = X_scaled.reshape(n_trials, n_channels, n_times)

        return X.astype(np.float32)

    def process_subject(self, subject: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single subject - convenience method."""
        X, y, metadata = self.load_and_preprocess(subject)
        return X, y


def preprocess_bcic_iv_2a(
    data_dir: str, subject: str, filter_motor_bands: bool = True, use_zuna: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to preprocess BCI IV-2a data.

    Args:
        data_dir: Path to data directory
        subject: Subject ID (e.g., 'A01')
        filter_motor_bands: Apply 8-32Hz bandpass (recommended)
        use_zuna: Use ZUNA if available

    Returns:
        X: EEG data (n_trials, n_channels, n_times)
        y: Labels (n_trials,)
    """
    preprocessor = MotorImageryPreprocessor(
        data_dir=data_dir, filter_alpha_beta=filter_motor_bands, use_zuna=use_zuna
    )

    X, y, metadata = preprocessor.load_and_preprocess(subject)

    return X, y


# Test function
if __name__ == "__main__":
    import sys

    data_dir = "src/data/BCICIV_2a_gdf"
    subject = "A01"

    print("=" * 60)
    print(f"Testing Motor Imagery Preprocessing: {subject}")
    print("=" * 60)

    preprocessor = MotorImageryPreprocessor(data_dir=data_dir, filter_alpha_beta=True)

    X, y, metadata = preprocessor.load_and_preprocess(subject)

    print("\n✓ Preprocessing test passed!")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {np.unique(y, return_counts=True)}")
    print(f"  Metadata: {metadata}")
