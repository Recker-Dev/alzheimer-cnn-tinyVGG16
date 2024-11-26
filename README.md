# CNN Model for Alzheimer's MRI Dataset

This project implements a Convolutional Neural Network (CNN) for classifying MRI scans into various stages of Alzheimer's Disease using the [Best Alzheimer's MRI Dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/data).

## Dataset

The dataset comprises high-quality MRI scans labeled for Alzheimer's Disease progression. It can be downloaded from [Kaggle](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/data).

### Problem Statement

The task is a **classification** problem to predict the stage of Alzheimer's Disease.

### Classes

The dataset includes the following classes:

- **Mild Impairment**
- **Moderate Impairment**
- **No Impairment**
- **Very Mild Impairment**

## Model Architecture

The CNN model comprises two convolutional blocks followed by a fully connected classifier; influenced by TinyVGG16 architecture.

## Performance Metrics

The model was evaluated on both training and test datasets. Below are the key metrics:

### Training Metrics

- **Loss:** 0.0039
- **Accuracy:** 99.90%
- **Precision:** 0.9990
- **Recall:** 0.9990
- **F1-Score:** 0.9990

### Testing Metrics

- **Loss:** 0.1574
- **Accuracy:** 95.47%
- **Precision:** 0.9563
- **Recall:** 0.9547
- **F1-Score:** 0.9548

### Train-Test Loss Over Epochs:

![Alt text](outputs\train-test-loss-over-epochs.png "Train Test Loss as observed over 20 epochs")

### Train Data Confusion Matrix:

![Alt text](outputs\train_data_cnf_mat.png "Train Data Confusion Matrix")

### Test Data Confusion Matrix:

![Alt text](outputs\test_data_cnf_mat.png "Test Data Confusion Matrix")

## How to get the code

```bash
git clone https://github.com/Recker-Dev/alzheimer-cnn-tinyVGG16.git
cd alzheimer-cnn
```

## How to Setup Environment

Follow these steps to set up and run the project:

1. **Create a Conda environment**  
   Run the following command to create a new environment named `alz_torch_cnn` with Python 3.10:

   ```bash
   conda create --name alz_torch_cnn python=3.10 -y
   ```

2. **Activate the environment**

   ```bash
   conda activate alz_torch_cnn
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Deactivate the environment**
   ```bash
   conda deactivate
   ```

## How to Use the Model

You can interact with the model in two ways:

1. **Using Jupyter Notebooks**  
   There are notebooks provided under the `notebooks` section. These notebooks allow you to experiment with the model, visualize predictions, and understand its workings in a step-by-step manner.

2. **Using the Streamlit Application**  
   For users without much coding background, an application (`app.py`) is provided for easy inference. Follow these steps to use the app:

   - Activate the environment:
     ```bash
     conda activate alz_torch_cnn
     ```
   - Run the Streamlit app:
     ```bash
     streamlit run app.py
     ```

3. **Testing Images**  
   Testing images are provided in the `Sample Testing Alzheimer Dataset` folder. Use these images to test the model's performance through the Streamlit app or the provided notebooks.

## Folder Structure:

```
root
├── app.py        # Main Streamlit application for user interaction
├── models
│   └── alz_CNN.pt  # Trained PyTorch model for Alzheimer's classification
├── model_arch.py  # Defines the architecture of the Alzheimer's classification model
├── notebooks      # Jupyter notebooks for development and analysis
│   ├── Alzehmier_CNN.ipynb  # Notebook for training the Alzheimer's classification model
│   └── predict_using_alzehmierCNN.ipynb  # Notebook for demonstrating model prediction
└── Sample Testing Alzehimer Dataset  # Folder containing sample images for testing
    └── test            # Subfolder containing images for inference/testing
        ├── Mild Impairment
        ├── Moderate Impairment
        ├── No Impairment
        └── Very Mild Impairment
└── requirements.txt  # File listing required Python libraries
```
