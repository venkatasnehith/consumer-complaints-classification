# consumer-complaints-classification
A Machine Learning model to classify consumer complaints into product categories using NLP and Logistic Regression

# 🗂️ Consumer Complaint Classification using NLP and Logistic Regression

This project classifies consumer complaints into predefined categories using Natural Language Processing (NLP) and Logistic Regression.

---

## 📌 Dataset

The dataset is downloaded from [KaggleHub](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp), which contains consumer complaints filed with the Consumer Financial Protection Bureau (CFPB).

- Each record contains:
  - **narrative** – the complaint text
  - **product** – the category of the complaint (label)

---

## 🔧 Technologies Used

- Python 🐍
- Google Colab 📓
- scikit-learn 🤖
- pandas, seaborn, matplotlib 📊
- TfidfVectorizer for feature extraction
- Logistic Regression for classification
- LabelEncoder for encoding class labels

---

## 🚀 How It Works

1. **Data is loaded** using `kagglehub`
2. **Preprocessing**: 
   - Remove nulls and empty strings
   - Clean text: keep alphabets only, convert to lowercase
3. **Label encoding** of product categories
4. **Text vectorization** using `TfidfVectorizer`
5. **Model training** using Logistic Regression
6. **Model evaluation** using accuracy and classification report
7. **Saving model & vectorizer** for reuse
8. **Prediction function** with a confidence threshold to detect unrelated inputs

---

## 🧪 Example

```python
Enter your complaint: I was charged twice for my credit card bill.
Predicted Category: credit_card
