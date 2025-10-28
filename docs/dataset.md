# Dataset Preparation

To train the DDSP for Guitar model, you need a dataset of clean electric guitar recordings. The quality and characteristics of your dataset will significantly impact the performance of the trained model.

## Data Requirements

- **Audio Format:** WAV format is recommended.
- **Sample Rate:** The model is designed for a 48kHz sample rate. If your data is at a different sample rate, you will need to resample it.
- **Clean Recordings:** The recordings should be as clean as possible, with minimal noise, reverb, or other effects. The goal is to capture the direct sound of the guitar.
- **Monophonic Lines:** The model is primarily designed for monophonic (single-note) lines. While it may work with chords to some extent, it's best to train on single-note recordings.
- **Variety:** A good dataset should cover a wide range of a guitar's capabilities:
    - **Pitch Range:** Cover the entire pitch range of the guitar.
    - **Dynamics:** Include notes played at various dynamic levels (from soft to loud).
    - **Articulations:** Include different playing styles such as staccato, legato, slides, bends, and vibrato.

## Preprocessing Steps

Once you have your dataset, you need to preprocess it to extract the necessary features for training. The main features required are:

1.  **Fundamental Frequency (f0):** This can be extracted using algorithms like CREPE or pYIN.
2.  **Loudness:** This can be calculated as the RMS energy of the audio signal.

The preprocessing pipeline would typically look like this:

1.  **Slice the audio:** Cut the long recordings into smaller segments (e.g., 4 seconds long).
2.  **Extract f0:** For each segment, extract the fundamental frequency contour.
3.  **Extract Loudness:** For each segment, calculate the loudness contour.
4.  **Save the features:** Save the audio segments and their corresponding f0 and loudness contours. It's common to save them in a format like `.npy` or `.hdf5` for efficient loading during training.

A helper script for preprocessing can be found in the `scripts/` directory (you may need to create this script).

### Using the Dummy Dataset Generator

For testing and development, a script is provided to generate a dummy dataset composed of simple sine waves. This allows you to run the training pipeline without needing a full dataset of guitar recordings.

To generate the dummy dataset, run the following command from the root of the repository:
```bash
python scripts/create_dummy_dataset.py --output_dir dataset
```
This will create a directory named `dataset` with the structure described below.

## Example Directory Structure

A preprocessed dataset might have a structure like this:

```
dataset/
├── train/
│   ├── audio/
│   │   ├── sample_0001.wav
│   │   ├── sample_0002.wav
│   │   └── ...
│   ├── f0/
│   │   ├── sample_0001.npy
│   │   ├── sample_0002.npy
│   │   └── ...
│   └── loudness/
│       ├── sample_0001.npy
│       ├── sample_0002.npy
│       └── ...
└── val/
    ├── audio/
    │   └── ...
    ├── f0/
    │   └── ...
    └── loudness/
        └── ...
```
