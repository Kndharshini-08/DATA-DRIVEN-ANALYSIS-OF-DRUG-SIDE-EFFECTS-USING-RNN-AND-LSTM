import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# Load data
df = pd.read_csv('Data/medicine_dataset.csv')

# Fill missing values
categorical_columns = ['Chemical Class', 'Therapeutic Class', 'Action Class']
df[categorical_columns] = df[categorical_columns].fillna('Unknown')
df['Habit Forming'] = df['Habit Forming'].fillna(df['Habit Forming'].mode()[0])
side_effect_columns = [col for col in df.columns if 'sideEffect' in col]
df[side_effect_columns] = df[side_effect_columns].fillna('')

# Preprocessing
def preprocess_data(data):
    features = ['Chemical Class', 'Therapeutic Class', 'Action Class', 'Habit Forming']
    X = data[features]
    X_encoded = pd.get_dummies(X)
    side_effect_columns = ['sideEffect0']
    y = data[side_effect_columns]
    return X_encoded, y

X, y = preprocess_data(df)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.values.ravel())

# Reshape X to be 3D for RNN input
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Convert y to categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build RNN model
model = Sequential()
model.add(SimpleRNN(50, input_shape=(1, X_train.shape[1]), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y_train_cat.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test_reshaped, y_test_cat))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_reshaped, y_test_cat)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Confusion Matrix
y_pred = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_cat, axis=1)

cm = confusion_matrix(y_test_classes, y_pred_classes)
print(cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, labels=np.unique(y_test_classes)))
