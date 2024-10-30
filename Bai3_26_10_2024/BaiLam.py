import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Chuẩn bị dữ liệu
# Tải tập dữ liệu hoa Iris (có thể thay bằng dữ liệu hình ảnh hoa, động vật từ tập dữ liệu của bạn)
data = datasets.load_iris()
X = data.data
y = data.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu (đặc biệt quan trọng với SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Hàm để huấn luyện và đánh giá mô hình
def evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Dự đoán và đánh giá
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    return train_time, accuracy, precision, recall


# Khởi tạo các mô hình
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# 3. Đánh giá và so sánh các mô hình
results = {}
for model_name, model in models.items():
    train_time, accuracy, precision, recall = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[model_name] = {
        'Time (s)': train_time,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

# 4. Hiển thị kết quả
for model_name, metrics in results.items():
    print(f"{model_name} Results:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print()
