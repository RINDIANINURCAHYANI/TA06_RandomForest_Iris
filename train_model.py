import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

# --- 1️⃣ Load dataset Iris ---
df = pd.read_csv('data/iris.csv', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print("Data berhasil dimuat!")
print(df.head())

# --- 2️⃣ Pisahkan fitur dan target ---
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# --- 3️⃣ Split data menjadi train dan test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 4️⃣ Latih Decision Tree ---
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# --- 5️⃣ Simpan model ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/decision_tree_model.pkl")
print("Model berhasil disimpan ke folder 'model/'")

# --- 6️⃣ Evaluasi model ---
predictions = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, predictions))

# --- 7️⃣ Pastikan folder images ada ---
os.makedirs("static/images", exist_ok=True)

# --- 8️⃣ Visualisasi Pohon Keputusan ---
plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True
)
plt.title("Pohon Keputusan - Dataset Iris")
plt.savefig("static/images/tree_visualization.png")
plt.close()

# --- 9️⃣ Confusion Matrix ---
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.title("Confusion Matrix - Decision Tree Iris")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.savefig("static/images/confusion_matrix.png")
plt.close()

# --- 🔟 Feature Importance ---
importances = model.feature_importances_
feature_df = pd.DataFrame({
    'Fitur': X.columns,
    'Pentingnya': importances
}).sort_values(by='Pentingnya', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Pentingnya', y='Fitur', data=feature_df, palette='viridis')
plt.title("Feature Importance - Decision Tree Iris")
plt.savefig("static/images/feature_importance.png")
plt.close()

# --- 11️⃣ ROC Curve ---
# Label binarize untuk multiclass
classes = model.classes_
y_test_bin = label_binarize(y_test, classes=classes)
y_pred_proba = model.predict_proba(X_test)

plt.figure(figsize=(8, 6))
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve - Decision Tree Iris")
plt.legend(loc="lower right")
plt.savefig("static/images/roc_curve.png")
plt.close()

print("\n✅ Model dan 4 visualisasi berhasil dibuat di folder static/images/")
