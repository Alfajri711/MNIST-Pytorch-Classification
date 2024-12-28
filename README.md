# MNIST Digit Classification with PyTorch

## Overview
This project demonstrates how to build, train, and evaluate a neural network for digit classification using the MNIST dataset. Leveraging PyTorch, the model achieves high accuracy, making it a great example of effective implementation of deep learning techniques.

## Features
- **Data Preparation**:
  - Loading and normalizing the MNIST dataset.
  - Visualizing dataset samples.
- **Model Architecture**:
  - Fully connected neural network.
  - ReLU activations and dropout layers for regularization.
- **Training and Evaluation**:
  - Training loop with cross-entropy loss and Adam optimizer.
  - Metrics such as accuracy, confusion matrix, and classification report.
- **Visualization**:
  - Displaying samples and results for better understanding.

## Model Details
- **Architecture**: Fully Connected Neural Network (FCNN)
  - **Input Layer**: Accepts 28x28 pixel images (flattened into a vector of size 784).
  - **Hidden Layers**: Multiple layers with ReLU activation functions.
  - **Regularization**: Dropout layers to prevent overfitting.
  - **Output Layer**: 10 units corresponding to digits (0-9) with softmax activation.
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: Adam Optimizer.

## Installation

### Prerequisites
Ensure you have Python 3.7+ and the following libraries installed:
- `torch`
- `torchvision`
- `matplotlib`
- `scikit-learn`
- `numpy`

You can install the required packages using pip:

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

### Clone the Repository

```bash
git clone https://github.com/username/mnist-pytorch-classification.git
cd mnist-pytorch-classification
```

## Usage

1. Open the Jupyter Notebook:

   ```bash
   jupyter notebook 04_03_Earth_Muhammad_Asri_Alfajri.ipynb
   ```

2. Run the cells sequentially to:
   - Prepare and visualize data.
   - Train the neural network model.
   - Evaluate model performance.

3. Modify hyperparameters or architecture to experiment further.

## Project Structure
```
.
├── data/               # MNIST data files (downloaded automatically)
├── notebooks/          # Jupyter Notebooks
├── requirements.txt    # Required Python packages
└── README.md           # Project documentation
```

## Results
- Achieved a test accuracy of **97.5%**.
- Generated a confusion matrix and detailed classification report to assess model performance.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
**Muhammad Asri Alfajri**

---

Feel free to reach out if you have any questions or suggestions!
