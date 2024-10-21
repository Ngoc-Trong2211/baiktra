from sklearn.model_selection import train_test_split
import pandas as pd
import re
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score 

# 1. Tải tập dữ liệu và chia thành tập huấn luyện (80%) và tập kiểm tra (20%)
data = pd.read_csv('Dataset-SA.csv')
data['Review'] = data['Review'].fillna('')  # Thay thế NaN bằng chuỗi rỗng

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 2. Thực hiện tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # Xóa dấu câu và ký tự đặc biệt
    return text

train_data['clean_text'] = train_data['Review'].apply(preprocess_text)
test_data['clean_text'] = test_data['Review'].apply(preprocess_text)

# 3. Sử dụng phương pháp TF-IDF để chuyển đổi văn bản thành dạng số
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(train_data['clean_text'])
X_test_tfidf = tfidf.transform(test_data['clean_text'])

# 4. Áp dụng thuật toán Naive Bayes để xây dựng mô hình phân loại
nb = MultinomialNB()
nb.fit(X_train_tfidf, train_data['Sentiment'])

# 5. Đánh giá mô hình bằng độ chính xác và F1-score
y_pred = nb.predict(X_test_tfidf)
accuracy = accuracy_score(test_data['Sentiment'], y_pred)
f1 = f1_score(test_data['Sentiment'], y_pred, average='weighted')

print(f'Do chinh xac: {accuracy}')
print(f'F1-score: {f1}')
