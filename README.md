# DDSP for Guitar

This project is an implementation of Differentiable Digital Signal Processing (DDSP) tailored for synthesizing electric guitar sounds. It utilizes a combination of harmonic synthesis, noise synthesis, waveshaping, and a tonestack to accurately model the sound of an electric guitar.

## Features

- **Harmonic Synthesis:** Generates the fundamental and harmonic frequencies of the guitar sound.
- **Noise Synthesis:** Models the noisy components of the sound.
- **Waveshaper:** A parametric tanh-based waveshaper to introduce non-linear distortion.
- **Tonestack:** A 3-band EQ (Low, Mid, High) to shape the tonal characteristics.
- **Transient Separator:** Separates the audio into transient and steady-state components for more detailed processing.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ddsp_guitar
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

Here is a basic example of how to load the model and synthesize audio.

```python
import torch
from ddsp_guitar.model import GuitarDDSP

# --- Configuration ---
SAMPLE_RATE = 48000
AUDIO_LENGTH_SECONDS = 4
FRAME_RATE = 250
HOP_SIZE = SAMPLE_RATE // FRAME_RATE

# --- Create dummy inputs ---
# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
audio_length_samples = AUDIO_LENGTH_SECONDS * SAMPLE_RATE
n_frames = audio_length_samples // HOP_SIZE

# Dummy audio input (e.g., from a microphone or file)
dummy_audio = torch.randn(batch_size, audio_length_samples).to(device)

# Dummy f0 and loudness (these would typically be extracted from the input audio)
dummy_f0_hz = torch.full((batch_size, n_frames), 110.0).to(device) # A2 note
dummy_loudness = torch.full((batch_size, n_frames), -30.0).to(device)

# --- Load Model ---
model = GuitarDDSP(sample_rate=SAMPLE_RATE).to(device)

# --- Synthesize Audio ---
# Note: The model's forward pass expects specific shapes.
# This is a simplified example.
# You will need to adapt your input tensors to the expected model dimensions.
# The audio input for the model's forward pass is typically a specific feature representation, not raw audio.
# For this example, we'll pass a tensor of the correct shape but it won't produce meaningful sound without proper conditioning.
conditioning_input = torch.randn(batch_size, n_frames).to(device)


# For a real use case, you would extract features from your `dummy_audio`
# and use them as input to the model's forward pass.
# The `forward` method in `model.py` expects `x`, `f0_hz`, and `loudness`.
# Let's assume `x` is the conditioning input.
output_audio = model(conditioning_input, dummy_f0_hz, dummy_loudness)

print("Synthesis complete!")
print("Output audio shape:", output_audio.shape)
```

For more detailed examples and tutorials, please refer to the documentation and the provided Google Colab notebooks.

## Generating a Dummy Dataset

To test the training pipeline without a full dataset, you can generate a dummy dataset of sine waves using the provided script:

```bash
python scripts/create_dummy_dataset.py --output_dir my_dummy_dataset
```

This will create a small dataset in the `my_dummy_dataset` directory with the required structure.
