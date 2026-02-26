

## 🩺 Disease Prediction System (Machine Learning)

This project focuses on predicting diseases based on patient symptoms using Machine Learning techniques. The dataset contains 100 records with features like Disease Name, Type, Symptoms, Doctor, and Remedies.

Symptoms were preprocessed using text cleaning and **TF-IDF vectorization** to convert them into numerical features. Multiple models were tested including **Naive Bayes, SVM, Logistic Regression, and Random Forest**.

Random Forest was implemented like a *team of doctors* — where each decision tree acts as a doctor giving an opinion, and the final prediction is based on collective decision-making. Hyperparameters such as `n_estimators`, `max_depth`, and `max_features` were tuned to improve performance.

Due to the small and imbalanced dataset, model accuracy remained limited, highlighting the importance of larger and balanced medical datasets for better prediction performance.

**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, TF-IDF, Matplotlib

---





Live server- https://angira-healthapi.onrender.com/

Images of prediction

<img width="519" height="283" alt="image" src="https://github.com/user-attachments/assets/1e49dae5-23ea-4772-9c99-f4f96721e4b1" />
<img width="1205" height="529" alt="image" src="https://github.com/user-attachments/assets/a1e553bc-9697-4e5c-b697-3f54ce17656a" />


