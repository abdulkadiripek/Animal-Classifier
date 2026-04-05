# Animal Image Classifier (Transfer Learning with MobileNetV2)

This repository contains an end-to-end image classification workflow built with PyTorch and torchvision.  
The current pipeline trains a multi-class animal classifier using transfer learning on top of MobileNetV2.

Although one notebook in the repository is named Facial_Emotion_Recognition.ipynb, the active training workflow shown in Animal-Classifier.ipynb is focused on classifying animal categories.

## Table of Contents

1. Project Overview  
2. Features  
3. Repository Structure  
4. Dataset and Class Labels  
5. Model Architecture  
6. Training and Evaluation Pipeline  
7. Inference Pipeline  
8. Requirements  
9. Environment Setup  
10. How to Run the Project  
11. Recommended Improvements  
12. Troubleshooting  
13. Reproducibility Notes  
14. FAQ  
15. Acknowledgments

## 1) Project Overview

The goal of this project is to build a reliable image classifier that can identify animal species/classes from RGB images. The project uses:

- PyTorch for model training and inference
- torchvision for datasets, transforms, and pretrained backbones
- MobileNetV2 pretrained weights for efficient transfer learning
- split-folders for train/test splitting

The high-level flow is:

1. Prepare dataset folders under animal_data.
2. Create train/test splits under dataset_split.
3. Build dataloaders with model-compatible transforms.
4. Load pretrained MobileNetV2 and replace the classifier head.
5. Train and evaluate.
6. Run prediction on custom images.

## 2) Features

- Transfer learning with a lightweight, fast architecture (MobileNetV2)
- Automatic preprocessing pipeline using pretrained model transforms
- Clear train/test loop implementation (loss + accuracy)
- Custom single-image prediction helper
- Notebook-based workflow for experimentation
- CPU/GPU device handling

## 3) Repository Structure

Below is the expected structure used by the notebook workflow:

```text
.
├── Animal-Classifier.ipynb
├── Facial_Emotion_Recognition.ipynb
├── README.md
├── animal_data/
│   ├── Bear/
│   ├── Bird/
│   ├── Cat/
│   ├── ...
│   └── Zebra/
└── dataset_split/
	 ├── train/
	 │   ├── Bear/
	 │   ├── Bird/
	 │   ├── ...
	 │   └── Zebra/
	 └── test/
		  ├── Bear/
		  ├── Bird/
		  ├── ...
		  └── Zebra/
```

Important note:

- dataset_split is generated data and is ignored in git (see .gitignore).
- Keep your raw source images in animal_data class folders.

## 4) Dataset and Class Labels

The notebook assumes a folder-per-class dataset where each class is a directory under animal_data.

Example classes in this repository include:

- Bear
- Bird
- Cat
- Cow
- Deer
- Dog
- Dolphin
- Elephant
- Giraffe
- Horse
- Kangaroo
- Lion
- Panda
- Tiger
- Zebra

The class order is discovered by torchvision.datasets.ImageFolder and stored as class_names.

## 5) Model Architecture

The model setup follows transfer learning best practices:

1. Load pretrained MobileNetV2 weights.
2. Freeze all backbone parameters.
3. Replace classifier with a new Linear layer sized to number of classes.
4. Train only the new classifier head (or optionally unfreeze later for fine-tuning).

Why MobileNetV2?

- Efficient and lightweight
- Strong pretrained features on ImageNet
- Good speed/accuracy trade-off for many practical tasks

## 6) Training and Evaluation Pipeline

The notebook defines:

- create_dataloader: builds train/test DataLoader objects
- train_step: one training epoch over the train dataloader
- test_step: one evaluation epoch over the test dataloader
- train: full epoch loop and metric logging

Tracked metrics:

- train_loss
- train_acc
- test_loss
- test_acc

Current defaults in notebook:

- Optimizer: AdamW
- Loss: CrossEntropyLoss
- Batch size: 128
- Epochs: 10

You can tune these based on your hardware and dataset size.

## 7) Inference Pipeline

A helper function (transform_predict) is used to run single-image inference:

1. Load image with PIL
2. Apply the same preprocessing transform used during training
3. Run forward pass in inference mode
4. Convert logits to probabilities using softmax
5. Show image with predicted class and confidence score

This is suitable for quick qualitative checks and demos.

## 8) Requirements

Main Python packages used:

- torch
- torchvision
- torchinfo
- split-folders
- Pillow
- matplotlib

Optional but recommended:

- jupyter
- ipykernel

## 9) Environment Setup

Create and activate a virtual environment, then install dependencies.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchinfo split-folders pillow matplotlib jupyter ipykernel
```

If you use CUDA-enabled PyTorch, install torch and torchvision from the official selector page so versions match your CUDA toolkit.

## 10) How to Run the Project

1. Place your dataset inside animal_data with one folder per class.
2. Open Animal-Classifier.ipynb.
3. Run cells in order:
	- imports
	- dataset split generation
	- dataloader creation
	- model setup
	- training loop
	- custom image prediction

If dataset_split already exists and you do not want to recreate it each run, skip the splitfolders cell.

## 11) Recommended Improvements

For better model quality and maintainability, consider:

- Add a validation split (train/val/test) and early stopping.
- Save best model checkpoints during training.
- Plot learning curves for loss and accuracy.
- Add confusion matrix and per-class precision/recall/F1.
- Introduce data augmentation (random crop, flip, color jitter).
- Perform staged fine-tuning (unfreeze last blocks after warm-up).
- Add deterministic seeding for reproducibility.
- Export model for deployment (TorchScript or ONNX).

## 12) Troubleshooting

Common issues and fixes:

1. RuntimeError: CUDA error or device mismatch  
	Ensure tensors and model are moved to the same device.

2. FileNotFoundError for dataset paths  
	Confirm animal_data and dataset_split folder names are correct.

3. ModuleNotFoundError for splitfolders or torchinfo  
	Install missing package with pip.

4. Notebook runs out of memory  
	Reduce batch size (for example 128 to 32 or 16).

5. Low accuracy  
	Check data quality, class balance, transform consistency, and train for more epochs.

## 13) Reproducibility Notes

For reproducible experiments:

- Set random seeds for Python, NumPy, and PyTorch.
- Keep library versions fixed in a requirements file.
- Save class_names with checkpoints.
- Log hyperparameters and final metrics per run.

## 14) FAQ

Q: Why does the repository include both Animal-Classifier and Facial_Emotion_Recognition notebooks?  
A: The active code shown here is an animal classifier workflow. If you plan to pivot to facial emotion recognition, adapt the dataset and labels accordingly.

Q: Can I train on CPU only?  
A: Yes, but it will be significantly slower than GPU training.

Q: Is dataset_split versioned in git?  
A: No. It is treated as generated data and excluded by .gitignore.

Q: Can I add new classes?  
A: Yes. Add a new class folder under animal_data and regenerate splits.

## 15) Acknowledgments

- PyTorch and torchvision teams for excellent tooling
- Open-source contributors for utilities like split-folders and torchinfo

If you want, this README can be extended with:

- A requirements.txt file
- A model checkpoint usage section
- Ready-to-run training script converted from the notebook
