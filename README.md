# 🧠 Alzheimer’s Disease Multi-Class MRI Classification

A deep learning project for classifying brain MRI images into four Alzheimer’s disease stages using **Transfer Learning (MobileNetV2)** with fine-tuning and data augmentation.

This project demonstrates a full machine learning pipeline: **EDA → Feature Analysis → Model Training → Fine-Tuning → Evaluation → Model Saving**, achieving **~90% test accuracy** on unseen data.

---

## 📌 Problem Statement

Early and accurate detection of Alzheimer’s disease is critical for patient care and treatment planning. Manual MRI interpretation is time-consuming and subjective. This project builds an automated multi-class classifier to identify disease stages from MRI images.

### Classification Categories

* **NonDemented**
* **VeryMildDemented**
* **MildDemented**
* **ModerateDemented**

---

## 📂 Dataset

**Source:** Kaggle – Alzheimer’s Disease Multiclass Dataset (Equal & Augmented)

The dataset contains preprocessed brain MRI images grouped into class-labeled folders.

### Dataset Structure

```
AlzheimerData/
├── train/
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
├── val/
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
```

---

## 🛠️ Tools & Technologies

* **Python 3.13**
* **TensorFlow / Keras**
* **MobileNetV2 (Pretrained on ImageNet)**
* **NumPy, Pandas**
* **Matplotlib, Seaborn**
* **Scikit-learn**

---

# 📊 Methodology

This project follows a structured, end-to-end machine learning workflow.

---

## 1️⃣ Exploratory Data Analysis (EDA)

### Class Distribution Analysis

We counted the number of images in each class to detect imbalance:

| Class            | Images  |
| ---------------- | ------- |
| MildDemented     | ~10,000 |
| ModerateDemented | ~10,000 |
| NonDemented      | ~12,800 |
| VeryMildDemented | ~11,200 |

### Observations

* The dataset is **slightly imbalanced**
* NonDemented and VeryMildDemented have more samples
* This can bias the model toward majority classes

### Solution

* Applied **class weighting** during training
* Used **data augmentation** to increase class variability

---

## 2️⃣ Image Quality Assessment

### Objective

Ensure MRI images are:

* Clear
* Consistent in contrast and brightness
* Free from distortions or corrupted files

### Approach

* Randomly sampled images from each class
* Visualized original images using Matplotlib
* Verified consistent brain-centered framing

### Outcome

* Images were uniform in resolution and quality
* No major noise or corrupted files detected

---

## 3️⃣ Feature Analysis (Traditional ML Baseline)

Before deep learning, statistical features were extracted:

* Mean pixel intensity
* Standard deviation
* Minimum pixel value
* Maximum pixel value

### Purpose

* Establish a baseline understanding of brightness and contrast differences
* Compare traditional ML performance vs deep learning

### Result

Traditional ML models achieved **~41% accuracy**, demonstrating that raw statistical features are insufficient for capturing complex brain structures.

---

## 4️⃣ Data Augmentation

### Why Augmentation?

Medical datasets are often limited and sensitive to overfitting. Augmentation helps the model:

* Generalize better
* Learn invariant features
* Avoid memorizing training samples

### Techniques Used

* Rotation
* Zoom
* Horizontal flipping
* Rescaling

### Validation

Brightness and histogram comparisons confirmed augmented images preserved anatomical realism without introducing artificial bias.

---

## 5️⃣ Transfer Learning Model Architecture

### Base Model

**MobileNetV2 (Pretrained on ImageNet)**

### Custom Layers

* Global Average Pooling
* Dense (ReLU)
* Dropout
* Softmax Output (4 Classes)

### Input Shape

```
224 × 224 × 3
```

---

## 6️⃣ Training Strategy

### Phase 1: Feature Extraction

* Base model frozen
* Only classification head trained

### Phase 2: Fine-Tuning

* Last 30 layers unfrozen
* Lower learning rate applied
* Allowed deeper layers to adapt to MRI-specific features

### Optimization

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Metrics: Accuracy

---

# 📈 Results

## 🔍 Traditional ML Baseline

| Metric   | Score |
| -------- | ----- |
| Accuracy | ~41%  |

### Interpretation

Statistical features alone cannot capture spatial brain patterns needed for Alzheimer’s classification.

---

## 🚀 Deep Learning Model Performance

### Test Set Evaluation

| Metric        | Value     |
| ------------- | --------- |
| Test Accuracy | **90.2%** |
| Test Loss     | 0.24      |

### Meaning

* The model generalizes well to unseen MRI scans
* Low loss indicates confident and stable predictions

---

## 📊 Classification Report

| Class            | Precision | Recall | F1-Score |
| ---------------- | --------- | ------ | -------- |
| MildDemented     | 0.93      | 0.92   | 0.92     |
| ModerateDemented | 0.99      | 1.00   | 0.99     |
| NonDemented      | 0.83      | 0.94   | 0.88     |
| VeryMildDemented | 0.90      | 0.76   | 0.83     |

### Insights

* **ModerateDemented** class achieved near-perfect performance
* Slight confusion between **VeryMildDemented and NonDemented**, which is expected due to visual similarity

---

## 🧩 Confusion Matrix Analysis

* Strong diagonal dominance indicates high correct classification
* Most errors occur between adjacent disease stages
* Confirms clinically realistic misclassification patterns

---

## 🔥 Fine-Tuning Impact

### Before Fine-Tuning

* Validation Accuracy: ~75–80%

### After Fine-Tuning

* Validation Accuracy: **~89–90%**

### Conclusion

Fine-tuning significantly improved:

* Feature specialization
* Class separation
* Overall generalization

---

# 💾 Model Saving

Final model saved in Keras format:

```
alzheimers_mobilenetv2_finetuned_v1.keras
```

Allows:

* Reuse
* Deployment
* Further training

---

# 📌 Future Improvements

* Add **Grad-CAM visualization** for explainability
* Train with **Vision Transformers (ViT)**
* Implement **cross-validation**
* Deploy as a web app (Flask / Streamlit)

---

# 🧠 Key Takeaways

* Transfer Learning dramatically outperforms traditional ML for medical imaging
* Fine-tuning enables domain-specific feature learning
* Data augmentation improves robustness and generalization
* Class weighting helps mitigate dataset imbalance

---

