# fruit shelf-life Estimation
A Multi-task Deep Learning model (MobileNetV2) that classifies fruits as Fresh/Rotten and simultaneously predicts their remaining shelf life in days. Built using PyTorch with a custom regression head for precise freshness estimation.
# Smart Fruit Shelf-Life Predictor (Multi-Task Learning)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)

## Project Overview
**A Multi-task Deep Learning model (MobileNetV2) that classifies fruits as Fresh/Rotten and simultaneously predicts their remaining shelf life in days.** This project solves a dual problem in food quality assurance:
1.  **Classification:** Identifying if a fruit is Fresh or Rotten.
2.  **Regression:** Estimating the remaining shelf-life (in days) based on visual features.

It uses a custom **Multi-Task Learning (MTL)** architecture where a shared backbone extracts features, which are then fed into two separate heads (one for classification, one for regression).

##  Key Features
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

##  Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Deployment/UI:** Gradio
* **Environment:** Google Colab / Python

##  Results
On the validation set, the model achieved:
* **Classification Accuracy:** ~99% (Distinguishing Fresh vs Rotten)
* **Regression MAE (Mean Absolute Error):** < 2.0 Days

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
*(You can add screenshots of your Gradio app or confusion matrix here)*

---
*Built  by Nimisha *
