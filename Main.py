#KHỞI TẠO
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

train_dir = r'C:\Users\ADMIN\Downloads\XLA\Data\dataset2-master\dataset2-master\images\TRAIN'
test_dir = r'C:\Users\ADMIN\Downloads\XLA\Data\dataset2-master\dataset2-master\images\TEST'

#LOAD DATA ẢNH
#KHỞI TẠO GENERATORS
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
   preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

#FLOW DATA ẢNH VÀO CÁC DIRECTORY
train_images = train_gen.flow_from_directory(    #80% of directory go to here
    directory=train_dir,
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)
val_images = train_gen.flow_from_directory(  #another 20% go to here
    directory=train_dir,
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=42,
    subset='validation'
)
test_images = test_gen.flow_from_directory(  
    directory=test_dir,
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=42
)

train_images.next()[1]

#Build Pretrained Model
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable = False

#Build Classification Model
inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(model.summary())

#Training
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=30,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

fig = px.line(
    history.history,
    y=['loss','val_loss'],
    labels={'index':"Epoch", "value": "Loss"},
    title="Training and Validation Loss Over Time"
)
fig.show()

#KẾT QUẢ
CLASS_NAMES = list(train_images.class_indices.keys())
CLASS_NAMES

predictions = np.argmax(model.predict(val_images), axis=1)

acc = accuracy_score(val_images.labels, predictions)
cm = tf.math.confusion_matrix(val_images.labels, predictions)
clr = classification_report(val_images.labels, predictions, target_names=CLASS_NAMES)

print("Validation Accuracy: {:.3f}%".format(acc * 100))

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks= np.arange(4) + 0.5, labels=CLASS_NAMES)
plt.yticks(ticks= np.arange(4) + 0.5, labels=CLASS_NAMES)
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Ma trận lỗi")
plt.show()

print("Classification Report:\n----------------------\n", clr)

model.save('model.h5')