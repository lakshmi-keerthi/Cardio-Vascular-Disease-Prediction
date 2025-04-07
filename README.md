# 🫀 Cardiovascular Disease Prediction using ML Models

This project uses machine learning algorithms to predict the likelihood of cardiovascular disease (CVD) based on patient medical and lifestyle data. With cardiovascular diseases being a leading cause of mortality globally, the objective is to assist in early detection through data-driven techniques.

https://cvduipy-munbl6pydqrfsgleoogwxt.streamlit.app/

![Sample Response](https://github.com/user-attachments/assets/331faa4f-f810-4c5e-baaa-a4ac3c16de18)  
![Sample Response](https://github.com/user-attachments/assets/91627c99-2bab-418e-aa5e-bda93cb8fb56)

---

## 📌 Project Overview

- **Dataset**: [Kaggle - Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Goal**: Predict whether a person is likely to develop cardiovascular disease based on 11 health metrics.
- **Data Size**: 70,000 records
- **Target Variable**: Presence or absence of cardiovascular disease (binary classification)

---

## 🛠️ Machine Learning Models

We implemented and compared multiple ML algorithms:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- Linear & Quadratic Discriminant Analysis

The **Gradient Boosting Classifier** gave the best results:
- **Accuracy**: ~73%
- **Sensitivity (Recall)**: ~69%
- **Best Params (GridSearchCV)**: `max_depth=5`, `n_estimators=90`

---

## 🔍 Key Steps

- **Exploratory Data Analysis (EDA)**: Identified trends, correlations, and outliers.
- **Outlier Removal**: Dropped lower and upper 5 percentile extremes to improve model performance.
- **Feature Engineering**: Standardized numerical features like age, weight, and blood pressure.
- **Data Scaling**: Used `StandardScaler` to normalize feature distributions.
- **Model Tuning**: Used GridSearchCV to find optimal hyperparameters.

- ![Flow Chart](https://github.com/user-attachments/assets/4b96cfb0-cd0f-4006-8065-659f2bd43b6b)


---

## 📊 Evaluation Metrics

| Model                       | Accuracy | Sensitivity | Objective Score |
|----------------------------|----------|-------------|-----------------|
| Logistic Regression        | 0.730    | 0.675       | 0.899           |
| Decision Tree              | 0.737    | 0.682       | 0.908           |
| Gradient Boosting Classifier | 0.736  | 0.695       | **0.910**       |
| K-Nearest Neighbors        | 0.737    | 0.677       | 0.906           |
| Random Forest              | 0.735    | 0.694       | 0.909           |

*Sensitivity (Recall) was prioritized due to the criticality of avoiding false negatives in medical predictions.*

---

## 👨‍💻 Team & Contributions

| Name                          | Role(s) |
|-------------------------------|---------|
| **Lakshmi Keerthi**           | Feature Engineering, EDA, Model Training & Evaluation, Documentation |
| **Indhu Sri Krishnaraj**      | Data Preprocessing, Visualization, Model Tuning |
| **Vishal Rachuri**            | Dataset Curation, Model Prediction & Evaluation |
| **Jyothika Vollireddy**       | Feature Selection, Algorithm Comparison |
| **Pavani Jaya Keerthana**     | Domain Understanding, Feature Engineering, Model Training |

---

## 📂 Code & Resources

- 📁 [Google Drive - Code Repository](https://drive.google.com/drive/folders/1KCAkxcnq9sbInlsl328ViC7d_sf-bt2M?usp=sharing)
- 📄 [Final Code PDF](https://drive.google.com/file/d/1ttfgfHnF301vxsYC075k47UcJY4d_bTV/view?usp=share_link)
- 📊 [Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

---

## 📚 References

- [Machine Learning Mastery: Outlier Detection](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/)
- [Gradient Boosting Explained](https://towardsdatascience.com/gradient-boosting-classification-explained-through-python-60cc980eeb3d)
- [Precision vs Recall](https://towardsdatascience.com/should-i-look-at-precision-recall-or-specificity-sensitivity-3946158aace1)

---

## 📌 Future Work

- Incorporate deep learning models for comparison.
- Extend to multiclass classification for different CVD severity levels.
- Integrate model into a lightweight web app for real-time predictions.

---

## 📬 Contact

For questions or collaboration:  
📧 LakshmiKeerthiGopireddy@my.unt.edu

---




