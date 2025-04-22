# 🧠 Customer Support Case Type Classification

This is a Python-based **Machine Learning** project that automatically classifies customer support cases into the following categories:  
✅ Billing  
✅ Technical  
✅ General  

---

## 🚀 Features

- ✅ Text classification using **Multinomial Naive Bayes**  
- ✅ Efficient **TF-IDF** vectorization for text features  
- ✅ Real-time prediction for new support cases  
- ✅ Evaluation using classification report and confusion matrix  
- ✅ Visual heatmap to interpret model performance  

---

## 🛠️ Installation

**Clone the repository:**

```bash
git clone https://github.com/YourUsername/Support-Case-Classifier.git
Navigate to the project folder:

bash
Copy
Edit
cd Support-Case-Classifier
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
📌 Usage
Ensure your dataset support_cases.csv is in the project directory.

Run the script:

bash
Copy
Edit
python classify_cases.py
You can:

See sample predictions

Test your own input messages

View evaluation metrics

💻 Code Overview
🧹 Data Preprocessing
python
Copy
Edit
df = df.dropna(subset=["case_description", "case_type"])
X = df["case_description"]
y = df["case_type"]
Cleans the dataset by removing missing values and separates features and labels.

📊 TF-IDF Vectorization
python
Copy
Edit
TfidfVectorizer(stop_words="english")
Converts text into numerical feature vectors.

🤖 Machine Learning Pipeline
python
Copy
Edit
Pipeline([
    ('tfidf', TfidfVectorizer(stop_words="english")),
    ('clf', MultinomialNB())
])
Combines TF-IDF with Naive Bayes for text classification.

📈 Evaluation Metrics
python
Copy
Edit
print(classification_report(y_test, y_pred))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
Displays model performance and visual confusion matrix.

🔮 Real-Time Prediction
python
Copy
Edit
user_input = input("Enter a support message: ")
predicted_label = clf_pipeline.predict([user_input])[0]
Test the classifier live with your own text input.

🌟 Example Output
✅ Case Prediction Example
bash
Copy
Edit
Enter a support message (or type 'exit' to quit): My subscription was charged twice.
The case type is likely: billing
📊 Evaluation Example
markdown
Copy
Edit
              precision    recall  f1-score   support

     billing       0.91      0.92      0.91       100
    technical     0.89      0.87      0.88       120
      general     0.86      0.87      0.86       130

    accuracy                           0.88       350
📝 Contributing
Feel free to fork the repository and submit a pull request. Any enhancements are welcome!

🏆 Acknowledgements
Built using:
🐍 Python
📘 Pandas
🔢 Scikit-learn
📊 Seaborn

Created with ❤️ by Aqdam

📃 License
This project is licensed under the MIT License.
See the LICENSE file for more details.

🔥 What’s Included:
✅ Clean format
✅ Installation and usage instructions
✅ Code overview
✅ Example output
✅ Contribution guidelines

yaml
Copy
Edit
