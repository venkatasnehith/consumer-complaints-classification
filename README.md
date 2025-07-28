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


```python
Enter your complaint: I didn't get a chocolate.
Predicted Category: ❓ Uncertain / Unrelated
```

---

## 🛠️ How to Use

```bash
!pip install kagglehub scikit-learn pandas matplotlib seaborn --quiet
```

Then run the notebook cell-by-cell on [Google Colab](https://colab.research.google.com).

---

## 📁 Files

* `complaint_classifier.pkl`: Trained Logistic Regression model
* `vectorizer.pkl`: TF-IDF Vectorizer
* `label_encoder.pkl`: Encoded labels
* `Colab_Notebook.ipynb`: Full notebook with code + explanation



----------------------------------------------------------------------------------------------------------------

## 📬 Contact

Created with 💡 by \[Your Name]

````

---

## ✅ Google Colab Notebook (Copy-Paste Ready)

I’ll now give you the **Colab notebook content** with **step-by-step comments** so you can understand it clearly. You can paste this into a new `.ipynb` or Colab notebook:

---

```python
# 📦 Install required packages
!pip install kagglehub scikit-learn pandas matplotlib seaborn --quiet
````

```python
# 🗂️ Download dataset using KaggleHub
import kagglehub

# Dataset: Consumer Complaint Dataset
path = kagglehub.dataset_download("shashwatwork/consume-complaints-dataset-fo-nlp")
print("✅ Dataset path:", path)
```

```python
# 📁 Explore the files
import os
import pandas as pd

print("📁 Files in dataset folder:")
for file in os.listdir(path):
    print(" -", file)
```

```python
# 📊 Load the CSV file into DataFrame
for file in os.listdir(path):
    if file.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(path, file))
            print("✅ Successfully loaded:", file)
            break
        except Exception as e:
            print("❌ Could not load:", file, "\n", e)
```

```python
# 🔍 Look at columns
df.columns
```

```python
# 🧹 Data Preprocessing
# Keep only 'product' and 'narrative' columns and clean
df = df[['product', 'narrative']].dropna()
df = df[df['narrative'].str.strip() != '']
df.rename(columns={'product': 'label', 'narrative': 'text'}, inplace=True)
```

```python
# 🧽 Clean text: remove special characters, lowercase everything
import re

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # keep letters
    text = text.lower()
    return text

df['clean_text'] = df['text'].apply(clean_text)
```

```python
# 🔢 Encode labels and split data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X = df['clean_text']
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
# 🔤 TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

```python
# 🧠 Train Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 📈 Evaluate the model
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
```

```python
# 💾 Save model, vectorizer, encoder
import joblib

joblib.dump(model, "complaint_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")
```

```python
# 🔄 Load them again (for prediction use)
model = joblib.load("complaint_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le = joblib.load("label_encoder.pkl")
```

```python
# 🔍 Prediction function with confidence check
def predict_complaint_category(complaint_text):
    import re
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        return text

    cleaned = clean_text(complaint_text)
    vect_text = vectorizer.transform([cleaned])
    
    probs = model.predict_proba(vect_text)[0]
    max_prob = max(probs)
    pred_class_index = probs.argmax()

    if max_prob < 0.5:
        return "❓ Uncertain / Unrelated"
    else:
        return le.inverse_transform([pred_class_index])[0]
```

```python
# 🎯 Try sample inputs
print(predict_complaint_category("I was charged a late fee even though I paid my loan on time."))
print(predict_complaint_category("I didn't get a chocolate"))
```

```python
# 📝 Interactive input in Google Colab
user_input = input("Enter your complaint: ")
predicted_label = predict_complaint_category(user_input)
print("Predicted Category:", predicted_label)
```

---


