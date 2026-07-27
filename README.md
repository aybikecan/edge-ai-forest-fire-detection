# Edge AI Forest Fire Detection

This project investigates efficient deep learning for image-based forest fire
detection in resource-constrained environments. It compares a custom CNN with
MobileNetV2 and EfficientNet-B0, applies dynamic quantization, and evaluates
edge-versus-cloud deployment trade-offs in latency, bandwidth, model size, and
estimated transmission energy.

## Project Overview

The notebook includes:

- Dataset download and inspection
- Exploratory data analysis and image-quality checks
- Image resizing, normalization, and data augmentation
- A custom CNN baseline
- Transfer learning with MobileNetV2 and EfficientNet-B0
- Classification metrics and confusion matrices
- Dynamic quantization for edge deployment
- Model-size, memory, and CPU-latency measurements
- Simulated edge-versus-cloud latency comparison
- Bandwidth and transmission-energy estimates
- A final single-image inference demonstration

## Dataset

The project uses the
[Forest Fire Dataset](https://www.kaggle.com/datasets/alik05/forest-fire-dataset)
from Kaggle. The data is downloaded at runtime and is not included in this
repository.

Please review the dataset page for its provenance, license, limitations, and
applicable terms of use.

## Models

- Custom baseline CNN
- Pretrained MobileNetV2
- Pretrained EfficientNet-B0

## Recorded Validation Results

| Model | Validation accuracy |
|---|---:|
| Baseline CNN | 0.9605 |
| MobileNetV2 | 0.9934 |
| EfficientNet-B0 | 0.9967 |

These results come from the notebook's recorded train-validation split and do not
represent performance on an independent real-world wildfire dataset.

## Edge Optimization Results

| Model | Accuracy | Size (MB) | CPU latency/image (ms) | Estimated memory (MB) |
|---|---:|---:|---:|---:|
| MobileNetV2 Original | 0.9934 | 8.72 | 51.14 | 8.62 |
| MobileNetV2 Quantized | 0.9934 | 8.72 | 49.53 | 8.61 |
| EfficientNet-B0 Original | 0.9967 | 15.58 | 61.24 | 15.46 |
| EfficientNet-B0 Quantized | 0.9967 | 15.57 | 64.24 | 15.45 |

The notebook selects **quantized MobileNetV2** as the preferred edge candidate
because it offers the lowest measured CPU latency with high validation accuracy
and a relatively small memory footprint.

Dynamic quantization is applied to linear layers only, so the reduction in total
model size is limited. Latency values are specific to the runtime used for the
recorded notebook execution and should be remeasured on the intended edge device.

## Edge-versus-Cloud Simulation

Under the notebook's assumptions:

- Average image size: 0.25 MB
- Upload speed: 5 Mbps
- Network delay: 200 ms
- Cloud inference latency: 80 ms

the estimated total latency is:

| Scenario | Total latency |
|---|---:|
| Edge inference | 49.53 ms |
| Cloud inference | 680.00 ms |

The bandwidth simulation estimates 4.22 GB of daily transfer for continuous cloud
monitoring versus 0.21 GB for edge monitoring. These are scenario-based estimates,
not measurements from a production deployment.

## Technologies

- Python
- PyTorch and Torchvision
- pandas and NumPy
- scikit-learn
- Matplotlib and seaborn
- Pillow
- Kaggle API
- Google Colab

## Running the Notebook

1. Open `Edge_AI_Forest_Fire_Detection.ipynb` in Google Colab.
2. Add the following values in Colab's **Secrets** panel:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
3. Enable notebook access for both secrets.
4. Run the notebook cells in order. A GPU runtime is recommended for training.

The notebook reads Kaggle credentials securely:

```python
from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
```

Never commit API keys, Kaggle credential files, downloaded datasets, or generated
model artifacts containing restricted data.

## Repository Structure

```text
edge-ai-forest-fire-detection/
├── Edge_AI_Forest_Fire_Detection.ipynb
└── README.md
```

## Limitations and Responsible Use

This project is an educational prototype, not a certified wildfire detection or
emergency-response system. Validation accuracy on a curated dataset does not
guarantee reliable operation under smoke, fog, poor lighting, unusual terrain,
camera faults, distribution shift, or previously unseen fire conditions.

False negatives could delay detection, while false positives could trigger
unnecessary alerts. Any real deployment would require representative field data,
external validation, calibrated decision thresholds, device-level benchmarking,
monitoring, redundancy, and human oversight.

