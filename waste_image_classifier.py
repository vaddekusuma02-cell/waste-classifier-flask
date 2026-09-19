import tensorflow as tf

dataset_path = r"C:\Users\vravi\Downloads\waste_dataset"

dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(180, 180),
    batch_size=16,
    validation_split=0.2,
    subset="training",
    seed=123
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(180, 180),
    batch_size=16,
    validation_split=0.2,
    subset="validation",
    seed=123
)

print("Classes:", dataset.class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    dataset,
    validation_data=validation_dataset,
    epochs=10
)

print("Training completed!")

model.save("waste_classifier.keras")

print("Model saved successfully!")