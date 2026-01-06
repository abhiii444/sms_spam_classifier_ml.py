# sms_spam_classifier_ml.py
SMS Spam Detection using NLP and Machine Learning. Includes data cleaning, EDA, TF-IDF vectorization, multiple classifiers, and ensemble techniques like Voting and Stacking for high-precision spam classification.
#  SMS Spam Classifier using Machine Learning & NLP

This project is an end-to-end **SMS Spam Detection System** built using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
It classifies SMS messages as **Spam** or **Ham (Not Spam)** with high accuracy and precision.

---

##  Project Highlights

- Complete ML pipeline from raw data to model evaluation
- Extensive Exploratory Data Analysis (EDA)
- Text preprocessing using NLP techniques
- Feature extraction with TF-IDF
- Comparison of multiple ML algorithms
- Ensemble learning using Voting and Stacking classifiers

---

##  Machine Learning Algorithms Used

- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Voting Classifier
- Stacking Classifier

---

##  Dataset Information

- **SMS Spam Collection Dataset**
- Target labels:
  - `0` → Ham (Not Spam)
  - `1` → Spam
- Dataset contains SMS messages with class imbalance handled during evaluation.

---

##  Tech Stack

- **Programming Language:** Python
- **Libraries & Tools:**
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - NLTK
  - Scikit-learn
  - XGBoost
  - WordCloud
  - TQDM

---

##  Project Workflow

1. Data Loading
2. Data Cleaning & Deduplication
3. Exploratory Data Analysis (EDA)
4. Text Preprocessing
   - Lowercasing
   - Tokenization
   - Stopword Removal
   - Stemming
5. Feature Engineering (TF-IDF Vectorization)
6. Model Training
7. Model Evaluation
8. Ensemble Learning

---

##  Evaluation Metrics

- Accuracy Score
- Precision Score
- Confusion Matrix

**Precision is prioritized** due to class imbalance in spam detection.

---

##  How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/abhiii444/sms-spam-classifier.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the script:
   ```bash
   python sms_spam_classifier_ml.py

---

## Results

- Multinomial Naive Bayes with TF-IDF achieved the best standalone performance.

- Voting and Stacking Classifiers further improved precision and robustness.

- The final model is highly effective for real-world spam filtering.

---

