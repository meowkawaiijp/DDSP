import os
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm

def create_dummy_dataset(output_dir='dataset', num_train=10, num_val=2, sr=48000, duration=4, frame_rate=250):
    """
    Generates a dummy dataset with sine waves for testing purposes.

    Args:
        output_dir (str): The root directory to save the dataset.
        num_train (int): The number of training samples to generate.
        num_val (int): The number of validation samples to generate.
        sr (int): The sample rate of the audio.
        duration (int): The duration of each audio clip in seconds.
        frame_rate (int): The frame rate for f0 and loudness features.
    """
    print(f"Creating dummy dataset in '{output_dir}'...")

    hop_size = sr // frame_rate
    num_samples = duration * sr
    num_frames = duration * frame_rate

    for split, num_files in [('train', num_train), ('val', num_val)]:
        print(f"Generating {split} set...")
        split_dir = os.path.join(output_dir, split)
        audio_dir = os.path.join(split_dir, 'audio')
        f0_dir = os.path.join(split_dir, 'f0')
        loudness_dir = os.path.join(split_dir, 'loudness')

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(f0_dir, exist_ok=True)
        os.makedirs(loudness_dir, exist_ok=True)

        for i in tqdm(range(num_files)):
            # --- Generate Audio (Sine Wave) ---
            freq = np.random.uniform(80.0, 800.0)  # Random frequency between E2 and G#5
            t = np.linspace(0., duration, num_samples)
            amplitude = np.random.uniform(0.2, 0.8)
            audio = amplitude * np.sin(2. * np.pi * freq * t)

            # --- Generate F0 and Loudness ---
            # For a simple sine wave, f0 is constant
            f0 = np.full(num_frames, freq, dtype=np.float32)

            # Calculate loudness (RMS energy)
            # Reshape audio into frames and calculate RMS
            audio_frames = np.lib.stride_tricks.as_strided(
                audio,
                shape=(num_frames, hop_size),
                strides=(audio.strides[0] * hop_size, audio.strides[0])
            )
            rms = np.sqrt(np.mean(np.square(audio_frames), axis=1))
            loudness = 20 * np.log10(rms + 1e-8) # Convert to dB

            # --- Save Files ---
            filename = f'sample_{i:04d}'
            sf.write(os.path.join(audio_dir, f'{filename}.wav'), audio, sr)
            np.save(os.path.join(f0_dir, f'{filename}.npy'), f0)
            np.save(os.path.join(loudness_dir, f'{filename}.npy'), loudness)

    print("Dummy dataset created successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a dummy dataset for DDSP Guitar.")
    parser.add_argument('--output_dir', type=str, default='dataset', help='Root directory for the generated dataset.')
    parser.add_argument('--num_train', type=int, default=10, help='Number of training samples.')
    parser.add_argument('--num_val', type=int, default=2, help='Number of validation samples.')
    parser.add_argument('--sr', type=int, default=48000, help='Sample rate.')
    parser.add_argument('--duration', type=int, default=4, help='Duration of each sample in seconds.')
    parser.add_argument('--frame_rate', type=int, default=250, help='Frame rate for features.')

    args = parser.parse_args()
    create_dummy_dataset(**vars(args))
