from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc

img_size = (256, 256)
batch_size = 32
num_classes = 5


test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    '5-classes/Test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale'
)

model = keras.models.load_model("X-ray-covid_model_3_v2.h5")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predict on the test set
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(class_labels)

# Print classification report
from sklearn.metrics import classification_report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Lấy nhãn thực tế của tập test
y_true = test_generator.classes  # Nhãn thực tế

# Dự đoán trên tập test
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred, average='weighted')  # Dùng 'macro' nếu cần trung bình đều
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
# Plot confusion matrix

conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Vẽ Precision-Recall Curve
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(true_classes == i, predictions[:, i])
    ap_score = average_precision_score(true_classes == i, predictions[:, i])
    plt.plot(recall, precision, label=f'Class {class_labels[i]} (AP = {ap_score:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend()
plt.show()

# Vẽ biểu đồ Accuracy per Class
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
plt.figure(figsize=(8, 6))
plt.bar(class_labels, class_accuracies, color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Accuracy')
plt.title('Accuracy per Class')
plt.ylim([0, 1])
plt.show()

# Vẽ biểu đồ cột Accuracy và Loss của mô hình
plt.figure(figsize=(6, 5))
plt.bar(['Accuracy', 'Loss'], [test_accuracy, test_loss], color=['green', 'red'])
plt.ylabel('Value')
plt.title('Model Performance Metrics')
plt.show()