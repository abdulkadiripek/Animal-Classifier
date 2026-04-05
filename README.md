# Animal Image Classifier

End-to-end multi-class animal image classification using transfer learning with PyTorch and MobileNetV2.

This repository trains a classifier that predicts animal categories from RGB images. The active workflow is implemented in `Animal-Classifier.ipynb`.

## Table of Contents

1. Project Goal
2. End-to-End Architecture
3. Data Layer
4. Model Layer
5. Training and Evaluation Flow
6. Inference Flow
7. Repository Structure
8. Tech Stack
9. Setup
10. How to Run
11. Results You Should Track
12. Design Decisions
13. Known Limitations
14. Recommended Next Steps
15. Troubleshooting
16. FAQ

## 1) Project Goal

The main objective is to build a practical, lightweight image classifier that can distinguish among multiple animal classes such as Bear, Cat, Dog, Elephant, Zebra, and others.

The project emphasizes:

- Transfer learning for fast convergence on limited data
- A reproducible folder-based dataset workflow
- A clear training loop (loss + accuracy tracking)
- Easy single-image prediction for demos

## 2) End-to-End Architecture

The pipeline is organized as a simple ML system with clear stages:

1. Raw images are stored per class in `animal_data/`.
2. The dataset is split into train/test (and optionally validation) under `dataset_split/`.
3. `ImageFolder` + `DataLoader` build batched tensors with model-compatible transforms.
4. A pretrained MobileNetV2 backbone is loaded.
5. The classification head is replaced to match the number of classes.
6. The model is trained with cross-entropy loss and AdamW.
7. Evaluation is performed each epoch on held-out data.
8. A prediction helper performs inference on new images.

High-level data flow:

```text
animal_data/ (raw class folders)
		  |
		  v
dataset_split/ (train/test[/val])
		  |
		  v
ImageFolder + Transforms + DataLoader
		  |
		  v
MobileNetV2 (pretrained backbone + custom classifier)
		  |
		  +--> Training loop (backprop, optimizer step)
		  |
		  +--> Evaluation loop (loss/accuracy)
		  |
		  v
Single-image inference (predicted class + confidence)
```

## 3) Data Layer

### 3.1 Dataset Format

The dataset uses a directory-per-class structure:

```text
animal_data/
  Bear/
  Bird/
  Cat/
  ...
  Zebra/
```

Class labels are automatically inferred by `torchvision.datasets.ImageFolder`.

### 3.2 Data Splitting Strategy

- Splitting is handled with `split-folders`.
- Output is generated under `dataset_split/`.
- `dataset_split/` is treated as generated data and should remain out of version control.

### 3.3 Input Preprocessing

Transforms are aligned with pretrained MobileNetV2 expectations (size/normalization). This keeps input distribution consistent with ImageNet-pretrained features and improves transfer performance.

## 4) Model Layer

### 4.1 Backbone

- Architecture: MobileNetV2
- Initialization: pretrained ImageNet weights
- Benefit: strong feature extractor with low compute cost

### 4.2 Transfer Learning Strategy

1. Freeze backbone parameters.
2. Replace final classifier layer with a new `Linear` layer sized to `num_classes`.
3. Train only the head first.
4. Optionally unfreeze deeper blocks later for fine-tuning.

This strategy gives stable training and good baseline accuracy without requiring large compute resources.

## 5) Training and Evaluation Flow

The notebook contains modular training utilities:

- `create_dataloader`: builds train/test dataloaders
- `train_step`: one epoch of forward + backward passes
- `test_step`: one evaluation epoch
- `train`: full multi-epoch orchestration and metric collection

Core configuration (current defaults):

- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW`
- Batch size: `128`
- Epochs: `10`

Tracked metrics:

- `train_loss`
- `train_acc`
- `test_loss`
- `test_acc`

## 6) Inference Flow

`transform_predict` runs single-image inference:

1. Load image with PIL.
2. Apply the same transform pipeline used in training.
3. Add batch dimension and move tensor to the selected device.
4. Run model in eval/inference mode.
5. Convert logits to probabilities with softmax.
6. Return top prediction and confidence score.

This provides a fast qualitative check for real images outside the dataset.

## 7) Repository Structure

```text
.
├── Animal-Classifier.ipynb
├── Facial_Emotion_Recognition.ipynb
├── README.md
├── animal_data/
│   ├── Bear/
│   ├── Bird/
│   ├── ...
│   └── Zebra/
└── dataset_split/
	 ├── train/
	 ├── test/
	 └── val/   (optional in your workflow)
```

Notes:

- `Animal-Classifier.ipynb` is the primary notebook for this animal classification pipeline.
- `Facial_Emotion_Recognition.ipynb` exists in the repository, but the current documented pipeline focuses on animals.

## 8) Tech Stack

- Python
- PyTorch
- torchvision
- Pillow
- matplotlib
- torchinfo
- split-folders
- Jupyter Notebook

## 9) Setup

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchinfo split-folders pillow matplotlib jupyter ipykernel
```

For GPU training, install CUDA-compatible `torch` and `torchvision` versions from the official PyTorch installation selector.

## 10) How to Run

1. Put your class folders inside `animal_data/`.
2. Open `Animal-Classifier.ipynb`.
3. Run cells in order:
	- imports and configuration
	- dataset split generation (skip if already prepared)
	- dataloader creation
	- model initialization
	- training/evaluation loop
	- single-image inference

## 11) Results You Should Track

At minimum:

- Final test accuracy
- Train vs test loss curves
- Class-wise errors (which classes are confused)

For stronger reporting:

- Confusion matrix
- Precision/Recall/F1 per class
- Best checkpoint epoch

## 12) Design Decisions

Why this architecture:

- MobileNetV2 keeps training and inference efficient.
- Transfer learning reduces data and compute requirements.
- Folder-based dataset structure makes adding classes simple.
- Notebook workflow accelerates experimentation and iteration.

## 13) Known Limitations

- No strict experiment tracking integrated yet.
- Validation split and early stopping may be missing depending on run setup.
- Performance can drop with class imbalance or low-quality images.
- Notebook-first workflow is less production-ready than a script/module layout.

## 14) Recommended Next Steps

1. Add deterministic seeding across Python/NumPy/PyTorch.
2. Add train/val/test separation with early stopping.
3. Save and load best checkpoints automatically.
4. Add confusion matrix and class-wise metrics.
5. Introduce stronger augmentation (flip/crop/color jitter).
6. Convert notebook pipeline to a reusable training script.
7. Export model (TorchScript or ONNX) for deployment.

## 15) Troubleshooting

1. `RuntimeError` about CUDA/device mismatch
	- Ensure model and tensors are on the same device.

2. `FileNotFoundError` for dataset paths
	- Verify `animal_data/` and `dataset_split/` paths.

3. `ModuleNotFoundError` for dependencies
	- Install missing packages with `pip`.

4. Out-of-memory during training
	- Reduce batch size (e.g., `128 -> 32` or `16`).

5. Low accuracy
	- Check label quality, class balance, augmentations, and epoch count.

## 16) FAQ

Q: Why is there also a `Facial_Emotion_Recognition.ipynb` file?

A: It is present in the repository, but the active documented workflow in this README is the animal classification pipeline.

Q: Can this run on CPU only?

A: Yes. It works on CPU, but training time will be significantly slower.

Q: Is `dataset_split/` committed to Git?

A: It should be treated as generated output and excluded from version control.

Q: How do I add a new class?

A: Add a new class folder under `animal_data/`, then regenerate the split and retrain.
