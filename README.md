# Malaria Cell Classification using Deep Learning

A PyTorch-based deep learning model for automated detection of malaria parasites in blood cell images using convolutional neural networks.

## 🎯 Project Overview

This project implements a custom CNN architecture to classify blood cell images as either **Parasitized** (infected with malaria) or **Uninfected**. The model achieves **94.30% test accuracy** on the malaria cell images dataset from Kaggle.

## 🏗️ Model Architecture

- **4 Convolutional Layers**: 32 → 64 → 128 → 256 filters
- **Max Pooling**: 2x2 pooling after each conv layer
- **3 Fully Connected Layers**: 512 → 128 → 2 neurons
- **Regularization**: 50% dropout to prevent overfitting
- **Activation**: ReLU activation functions
- **Input Size**: 224x224 RGB images

## 📊 Dataset

- **Source**: [Kaggle - Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- **Total Images**: 27,558 cell images
- **Classes**: 
  - Parasitized: 13,779 images
  - Uninfected: 13,779 images
- **Split**: 70% training, 30% testing

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision kaggle matplotlib seaborn scikit-learn
```

### Setup Kaggle API

1. Create a Kaggle account and generate API token
2. Place `kaggle.json` in `~/.kaggle/` directory
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Run the Model

```bash
python cellclassificationmodel.py
```

The script will automatically:
1. Download the dataset from Kaggle
2. Split data into train/test sets
3. Train the CNN model
4. Evaluate performance and generate confusion matrix

## 📈 Results

- **Training Accuracy**: 85.85% (after 5 epochs)
- **Test Accuracy**: 94.30%
- **Test Loss**: 0.1558

### Training Progress
```
Epoch 1: Train Acc: 78.57%, Train Loss: 0.4833
Epoch 2: Train Acc: 84.40%, Train Loss: 0.3723
Epoch 3: Train Acc: 84.79%, Train Loss: 0.3552
Epoch 4: Train Acc: 85.72%, Train Loss: 0.3442
Epoch 5: Train Acc: 85.85%, Train Loss: 0.3383
```

## 🔧 Technical Features

- **Automated Data Pipeline**: Kaggle API integration for seamless dataset download
- **Data Augmentation**: Random resized crop and horizontal flip
- **GPU Acceleration**: MPS (Metal Performance Shaders) support for Mac
- **Visualization**: Confusion matrix generation with Seaborn
- **Model Persistence**: Automatic model saving capabilities

## 📁 Project Structure

```
malaria-detection/
├── cellclassificationmodel.py    # Main training script
├── README.md                     # Project documentation
├── data/                         # Dataset directory (auto-created)
│   └── malaria/
│       ├── train/               # Training images
│       ├── test/                # Test images
│       └── cell_images/         # Original dataset
├── confusion_matrix.png         # Generated confusion matrix
└── model.pth                    # Saved model weights
```

## 🧠 Model Details

### Data Preprocessing
- Images resized to 224x224 pixels
- Normalization with ImageNet statistics
- Random augmentation for training data

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 5
- **Device**: MPS (Mac GPU) / CPU fallback

## 🔬 Medical Impact

This model can assist healthcare professionals in:
- **Rapid malaria screening** in resource-limited settings
- **Reducing diagnostic time** from hours to seconds
- **Supporting telemedicine** applications
- **Training medical students** with consistent examples

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements:
- Model architecture enhancements
- Additional data augmentation techniques
- Performance optimizations
- Documentation improvements

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset provided by the National Library of Medicine (NIH)
- Kaggle for hosting the dataset
- PyTorch team for the deep learning framework

---

**Note**: This model is for research and educational purposes. For clinical applications, please consult with medical professionals and follow appropriate regulatory guidelines.
