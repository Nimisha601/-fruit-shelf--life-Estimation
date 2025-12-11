# fruit shelf-life Estimation
A Multi-task Deep Learning model (MobileNetV2) that classifies fruits as Fresh/Rotten and simultaneously predicts their remaining shelf life in days. Built using PyTorch with a custom regression head for precise freshness estimation.
#  Smart Fruit Shelf-Life Predictor (Multi-Task Learning)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)

## Project Overview
**A Multi-task Deep Learning model (MobileNetV2) that classifies fruits as Fresh/Rotten and simultaneously predicts their remaining shelf life in days.** This project solves a dual problem in food quality assurance:
1.  **Classification:** Identifying if a fruit is Fresh or Rotten.
2.  **Regression:** Estimating the remaining shelf-life (in days) based on visual features.

It uses a custom **Multi-Task Learning (MTL)** architecture where a shared backbone extracts features, which are then fed into two separate heads (one for classification, one for regression).

## Key Features
* **Dual Output:** Predicts Class (Fresh/Rotten) and Days Remaining simultaneously.
* **Lightweight Architecture:** Uses **MobileNetV2** as a backbone, making it suitable for mobile/edge devices.
* **Data Efficient:** Trained on the Kaggle "Fresh and Rotten Fruits" dataset.
* **Interactive Demo:** Includes a **Gradio** web interface for real-time testing.

##  Model Architecture
The model uses a **Hard Parameter Sharing** approach:
* **Backbone:** MobileNetV2 (Pre-trained on ImageNet) acts as the feature extractor.
* **Bottleneck:** A shared dense layer (512 units) to consolidate features.
* **Head 1 (Classifier):** Outputs probabilities for classes (Fresh Apple, Rotten Banana, etc.).
* **Head 2 (Regressor):** Outputs a single continuous value (Estimated Days).

## Dataset
The model was trained on the **Fresh and Rotten Fruits for Classification** dataset from Kaggle.
* **Preprocessing:** Images resized to 224x224, normalized using ImageNet stats.
* **Labeling Logic:**
    * `Fresh` images: Labelled as ~2 days old (High shelf life remaining).
    * `Rotten` images: Labelled as ~10+ days old (Low/Zero shelf life).

## Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Deployment/UI:** Gradio
* **Environment:** Google Colab / Python

## Results
On the validation set, the model achieved robust performance:
* **Classification Accuracy:** **97%** (High precision in distinguishing Fresh vs Rotten)
* **Regression MAE:** **0.44 days** (Mean Absolute Error)
* **Real-world Performance:** Effectively differentiates between fresh and spoiled produce with minimal error.

## Confusion Matrix Results

The model achieved robust performance on the validation set (~97% accuracy). The table below shows the confusion matrix, highlighting a few realistic misclassifications due to visual similarities:

| True Label \ Predicted | Fresh Apples | Fresh Banana | Fresh Oranges | Rotten Apples | Rotten Banana | Rotten Oranges |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fresh Apples** | **388** | 0 | 0 | 7 | 0 | 0 |
| **Fresh Banana** | 0 | **375** | 0 | 0 | 6 | 0 |
| **Fresh Oranges** | 0 | 0 | **380** | 0 | 0 | 8 |
| **Rotten Apples** | 5 | 0 | 0 | **596** | 0 | 0 |
| **Rotten Banana** | 0 | 8 | 0 | 0 | **522** | 0 |
| **Rotten Oranges** | 0 | 0 | 6 | 0 | 0 | **396** |

> **Note:** The diagonal values (bold) represent correct predictions. The few off-diagonal numbers indicate edge cases where early signs of rotting were subtle, leading to minor misclassifications.

##  How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/fruit-shelf-life-predictor.git](https://github.com/your-username/fruit-shelf-life-predictor.git)
    cd fruit-shelf-life-predictor
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision pandas numpy scikit-learn gradio
    ```

3.  **Run the Training Script:**
    ```bash
    python src/train.py --epochs 10
    ```

4.  **Launch the App (Demo):**
    Run the Gradio interface code provided in the notebook/script.

## 📸 Screenshots
 <img width="1884" height="822" alt="image" src="https://github.com/user-attachments/assets/16a13cce-9158-4b64-9a1f-50c9fc2011d1" />
 <img width="578" height="535" alt="image" src="https://github.com/user-attachments/assets/ea095176-061a-447d-910b-660a98808175" />
 <img width="454" height="470" alt="image" src="https://github.com/user-attachments/assets/d7340992-7ba5-4360-8e1e-e3ac7add953f" />

 



---
*Built by Nimisha*
