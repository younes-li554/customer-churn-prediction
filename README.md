
# Customer Churn Prediction

This repository contains a **Machine Learning project** for predicting customer churn for a telecom company. The project demonstrates the complete workflow from **data preprocessing, exploratory data analysis, model training, model explainability, model saving, API deployment, to creating a simple dashboard**.

---

## 📂 Dataset

The dataset used is the **Telco Customer Churn dataset**, which contains customer information and whether they have churned (left the service) or not. Key columns include:

- `customerID`: Unique customer identifier.
- `gender`: Customer gender (Male/Female).
- `SeniorCitizen`: Indicates if the customer is a senior (1 = Yes, 0 = No).
- `Partner`, `Dependents`: Customer's family status.
- `tenure`: Number of months the customer has stayed with the company.
- `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: Service-related features.
- `Contract`: Contract type (Month-to-month, 1-year, 2-year).
- `PaperlessBilling`: Whether the customer uses paperless billing.
- `PaymentMethod`: Payment method (Credit card, Bank transfer, etc.).
- `MonthlyCharges`, `TotalCharges`: Customer billing amounts.
- `Churn`: Target variable indicating if the customer left the service (Yes/No).

The dataset is stored as a CSV file and loaded using `pandas`.

---

## 🔍 Data Exploration & Visualization

We first explore the data to understand its structure, missing values, and distribution of key variables:

- Checked for missing and duplicated values and handled them appropriately.
- Converted categorical text columns (`Yes`/`No`) to numeric binary values.
- Scaled numeric features to normalize their range.
- Used **visualizations** (Matplotlib, Seaborn, Plotly) to analyze:
  - Churn distribution
  - Churn by gender, contract type, dependents, senior citizenship, and services
  - Billing amounts distribution and tenure

These steps helped us understand the patterns in the data and informed feature engineering decisions.

---

## ⚙️ Data Preprocessing

Preprocessing steps include:

1. Dropping irrelevant columns such as `customerID`.
2. Handling missing values (`TotalCharges` conversion to numeric and filling NAs).
3. Encoding categorical variables using **one-hot encoding**.
4. Splitting the dataset into **training and test sets**.
5. Scaling numeric features using `MinMaxScaler`.

---

## 🤖 Model Training

We implemented several machine learning models using a **pipeline**:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Classifier (SVC)**
- **Random Forest Classifier**

**Hyperparameter tuning** was performed using `GridSearchCV` with 5-fold cross-validation. The final model was selected based on the **best accuracy on the validation set**.

Evaluation metrics include:

- Accuracy
- Confusion matrix
- Classification report
- ROC curve and AUC score

---

## 🧩 Model Explainability with SHAP

We used **SHAP (SHapley Additive exPlanations)** to interpret the model predictions:

- SHAP values quantify the impact of each feature on a prediction.
- We generated:
  - **Summary plots** showing global feature importance
  - **Force plots** for individual predictions to see which features push the prediction towards churn or not

This step helps stakeholders understand **why the model predicts a customer will churn**, increasing trust in the model.

---

## 💾 Saving the Model

The final trained model and feature columns are saved using `joblib`:

```python
joblib.dump(best_model, "churn_model.pkl")
joblib.dump(X_train.columns, "X_train_columns.pkl")
````

This allows the model to be loaded later for predictions without retraining.

---

## 🌐 Model Deployment (API)

The project includes a **FastAPI endpoint** to serve predictions:

* Accepts customer data in JSON format
* Aligns input features with training columns
* Returns **churn probability** and **churn prediction**

Example endpoint usage:

```json
POST /predict
{
  "tenure": 12,
  "MonthlyCharges": 70,
  "TotalCharges": 840,
  "gender_Female": 0,
  ...
}
```

Response:

```json
{
  "Churn Probability": 0.23,
  "Churn Prediction": 0
}
```

The API allows integration with other applications for real-time predictions.

---

## 📊 User Interface / Dashboard (Optional)

We also created a simple **Streamlit dashboard**:

* Users can input customer information via a sidebar
* Dashboard displays predicted churn probability and prediction
* Works interactively and provides a visual way to test different customer profiles

```bash
streamlit run dashboard.py
```

**Note:** Streamlit must be installed, and the app should be run outside the notebook environment (e.g., terminal).

---

## 🛠️ Libraries Used

* **Data Handling**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`, `plotly`, `shap`
* **Preprocessing & Models**: `scikit-learn`
* **Model Saving**: `joblib`
* **API Deployment**: `fastapi`, `pydantic`
* **Dashboard**: `streamlit` (optional)

---

## 📌 How to Run

1. Clone the repository
2. Install the required libraries
3. Run the notebook for data exploration and model training
4. Use `joblib` files to deploy API or Streamlit dashboard

---

## 📖 Summary

This project demonstrates the **end-to-end machine learning workflow**:

* Data exploration and preprocessing
* Model training and evaluation
* Model explainability using SHAP
* Model saving for reuse
* API deployment for predictions
* Optional interactive dashboard

It serves as a template for **predictive modeling projects in telecom or other industries**.

```

---

If you want, I can also make a **shorter, GitHub-ready version** with badges, table of contents, and links so it looks professional on the repository front page.  

Do you want me to do that?
```
