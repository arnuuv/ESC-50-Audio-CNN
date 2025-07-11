# Audio-CNN-from-scratch

This project implements an audio classification CNN from scratch using PyTorch and torchaudio. It is designed to run both locally and on Modal for scalable inference and training.

## Sample UI Screenshots

Below are example screenshots of the CNN Audio Visualizer web UI:

### Top Predictions, Spectrogram, and Waveform

![UI showing top predictions, input spectrogram, and audio waveform](./audio-cnn-visualisation/sample-ui-1.jpg)

### Convolutional Layer Outputs

![UI showing convolutional layer outputs and feature maps](./audio-cnn-visualisation/sample-ui-2.jpg)

## Features

- Residual CNN architecture for audio classification
- ESC-50 dataset support
- Modal integration for cloud training and inference
- Feature map visualization support

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## Model File

- The trained model should be named `best_models.pth`.
- When running on Modal, this file must be present in the root of the `esc-model` Modal volume.
- For local runs, download `best_models.pth` to your project directory.

## Usage

### 1. Local Inference (`modal run`)

1. Download or place your trained model as `best_models.pth` in the project root.
2. Place your test audio file (e.g., `opening-soda-can.wav`) in the project root.
3. Run:
   ```bash
   modal run main.py
   ```
   This will print the top predictions and waveform info for the audio file.

### 2. Cloud Inference (`modal deploy`)

1. Ensure your model is uploaded to the `esc-model` Modal volume as `best_models.pth`.
2. Deploy the app:
   ```bash
   modal deploy main.py
   ```
3. Use the provided web endpoint to send POST requests with base64-encoded audio data for inference.

### 3. Training

- See `train.py` for training logic. The model expects 22050 Hz audio (the default in the code).

## Notes

- The model expects audio to be resampled to 22050 Hz for both training and inference.
- The main entrypoint now uses `opening-soda-can.wav` as the default test file.
- For feature map visualization, ensure your model and inference code support the `return_feature_maps` argument.

## Example Output

```
Top predictions:
  -<class1> <confidence1>
  -<class2> <confidence2>
  -<class3> <confidence3>
First 10 values: [0.0, 0.1, ...]
Duration: 1.23
```

## Troubleshooting

- If you get a `FileNotFoundError` for the model, ensure `best_models.pth`
