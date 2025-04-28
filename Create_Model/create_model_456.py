import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D,  BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, models

# Kiểm tra và sử dụng GPU CUDA
gpus = tf.config.experimental.list_physical_devices('GPU')
print("🔍 GPU CUDA Devices:", gpus)
if gpus:
    try:
        if len(gpus) > 1:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')  # Chọn GPU thứ 2 (ID 1)
            tf.config.experimental.set_memory_growth(gpus[1], True)  # Tránh lỗi OOM
            print("✅ Đã sử dụng GPU:", gpus[1])
        else:
            print("⚠ Chỉ có một GPU khả dụng:", gpus[0])
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  # Chọn GPU đầu tiên (ID 0)
            tf.config.experimental.set_memory_growth(gpus[0], True)  # Tránh lỗi OOM
    except RuntimeError as e:
        print("⚠ Lỗi khi bật GPU:", e)
else:
    print("⚠ Không có GPU khả dụng")

def plot_images(images, titles, ncols=5):
    nrows = (len(images) + ncols - 1) // ncols
    plt.figure(figsize=(ncols * 2, nrows * 2))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Cấu hình dữ liệu đầu vào
img_size = (256, 256)
batch_size = 32
num_classes = 5


# Chuẩn bị dữ liệu với ImageDataGenerator
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Tải dữ liệu từ thư mục
train_generator = train_datagen.flow_from_directory(
    'D:/Deep_learing/Do_an/data/Train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

val_generator = val_datagen.flow_from_directory(
    'D:/Deep_learing/Do_an/data/Val',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# Get a batch of images
images, labels = next(train_generator)

print(f"Số lượng ảnh sau khi tăng cường: {train_generator.n}")

# Plot preprocessed images
plot_images(images[:10], [f'Label: {np.argmax(label)}' for label in labels[:10]])

def visualize_activations(model, images):
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(images)

    for layer_name, layer_activation in zip([layer.name for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name], activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // 16
        display_grid = np.zeros((size * n_cols, size * 16))
        for col in range(n_cols):
            for row in range(16):
                channel_image = layer_activation[0, :, :, col * 16 + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='gray')


# Xây dựng mô hình CNN
def create_model_1():
    model = Sequential()
    input_shape = (256, 256, 1)
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization()),
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization()),
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2)),
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2)),
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# def create_model_2():
#     model = Sequential()
#     input_shape = (256, 256, 1)
#
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.4))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.4))
#     model.add(Dense(num_classes, activation='softmax'))
#     # Softmax cho phân loại đa lớp
#
#     optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
#
#     return model

# def create_model_2_1():
#     model = Sequential()
#     input_shape = (256, 256, 1)
#
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.3))
#
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
#
#     return model

# def create_model_3():
#     model = Sequential()
#
#
#     model.add(
#         Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001), input_shape=(224, 224, 1)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.3))  # Chống overfitting
#
#
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.3))
#
#
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.4))
#
#
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.5))
#
#
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(Dropout(0.5))  # Dropout mạnh
#     model.add(Dense(num_classes, activation='softmax'))  # Softmax cho phân loại đa lớp
#
#
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Learning rate nhỏ giúp ổn định
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
#     return model



def create_model_3():

    model = Sequential()
    input_shape = (256, 256, 1)
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# def create_model_4():
#     input_shape = (256, 256, 1)
#     model = models.Sequential([
#         # Block 1
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#
#         # Block 2
#         layers.Conv2D(64, (3, 3), activation='relu',),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#
#         # Block 3
#         layers.Conv2D(128, (3, 3), activation='relu'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#
#         # Block 4
#         layers.Conv2D(256, (3, 3), activation='relu',),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
#
#         # Flatten + Dense
#         layers.Flatten(),
#         layers.Dense(512, activation='relu'),
#         layers.Dropout(0.5),  # Giảm overfitting
#         layers.Dense(num_classes, activation='softmax')  # Phân loại 5 lớp
#     ])
#
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model


#Khởi tạo mô hình
# model_1 = create_model_1()
# model_1.summary()
# #
# # model_2 = create_model_2()
# # model_2.summary()
# #
# # model_3 = create_model_3()
# # model_3.summary()
#
# # model_4 = create_model_4()
# # model_4.summary()
# # model_2 = create_model_2_1()
# # model_2.summary()

model_3 = create_model_3()
model_3.summary()

#================================================================================================

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]
# Huấn luyện mô hình
history = model_3.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks
)
visualize_activations(model_3, images[:1])

model_3.save("X-ray-covid_model_3_v2.h5")
print("✅ Mô hình đã được lưu: X-ray-covid_model_3_v2.h5")

# Dự đoán trên tập validation
y_pred = model_3.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes
#================================================================================================

# Vẽ biểu đồ Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# Vẽ biểu đồ Loss (Mất mát)
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()


# Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


