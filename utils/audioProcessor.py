# utils/audio_processor.py
import librosa
import numpy as np
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, target_sr: int = 16000, top_db: int = 25):
        self.target_sr = target_sr
        self.top_db = top_db
        
    def load_and_trim_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and trim silence
        Args:
            file_path: Path to audio file
        Returns:
            Tuple of (trimmed_audio_array, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=self.top_db)
            return audio_trimmed, sr
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {str(e)}")
            raise

    def extract_advanced_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract comprehensive audio features
        Args:
            audio: Audio signal array
            sr: Sampling rate
        Returns:
            Combined feature vector (numpy array)
        """
        try:
            # Spectral features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            
            # Temporal features
            tempogram = librosa.feature.tempogram(y=audio, sr=sr)
            
            # Feature aggregation
            features = np.vstack([
                mfcc,
                chroma,
                spectral_contrast,
                tonnetz,
                tempogram
            ])
            
            # Statistical aggregation
            return self._aggregate_features(features)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            raise

    def _aggregate_features(self, features: np.ndarray) -> np.ndarray:
        """
        Aggregate features using statistical measures
        Args:
            features: Raw feature matrix
        Returns:
            Flattened feature vector
        """
        mean = np.mean(features, axis=1)
        std = np.std(features, axis=1)
        percentiles = np.percentile(features, [25, 50, 75], axis=1)
        return np.concatenate([mean, std, percentiles.flatten()])

    def plot_waveform(self, audio: np.ndarray, sr: int) -> Optional[plt.Figure]:
        """
        Generate waveform plot using matplotlib
        Args:
            audio: Audio signal array
            sr: Sampling rate
        Returns:
            matplotlib Figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(audio, sr=sr, ax=ax)
            ax.set_title('Audio Waveform')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            return fig
        except Exception as e:
            logger.error(f"Waveform plotting error: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    processor = AudioProcessor()
    audio, sr = processor.load_and_trim_audio("sample.wav")
    features = processor.extract_advanced_features(audio, sr)
    print(f"Extracted features shape: {features.shape}")