# 匯入所需的庫
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 載入 MNIST 數據集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 正規化數據
train_images, test_images = train_images / 255.0, test_images / 255.0

# 創建神經網絡模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 將 28x28 的影像展平
    layers.Dense(128, activation='relu'),  # 隱藏層，使用 ReLU 激活函數
    layers.Dropout(0.2),  # 隨機失活，避免過擬合
    layers.Dense(10, activation='softmax')  # 輸出層，使用 softmax 激活函數
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(train_images, train_labels, epochs=5)

# 測試模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'\nTest accuracy: {test_acc}')

# 使用模型進行預測
predictions = model.predict(test_images)

# 顯示預測結果
plt.figure(figsize=(10, 10))
for i in range(25):
    # 選擇不同的索引以使用不同的測試圖片
    index = i  # 這裡可以修改為你想要的不同索引
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 使用不同的測試圖片
    plt.imshow(test_images[index], cmap=plt.cm.binary)
    predicted_label = tf.argmax(predictions[index])
    true_label = test_labels[index]
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel(f'Predicted: {predicted_label}, True: {true_label}', color=color)

plt.show()
