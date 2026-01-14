import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_DIR = './data'
data = []
labels = []

# 1. Load data from folders
for dir_ in os.listdir(DATA_DIR):
    for path in os.listdir(os.path.join(DATA_DIR, dir_)):
        with open(os.path.join(DATA_DIR, dir_, path), 'r') as f:
            line = f.readline()
            landmarks = [float(x) for x in line.split(',')]
            data.append(landmarks)
            labels.append(dir_)

X = np.asarray(data)
y = np.asarray(labels)

# 2. Split data: 80% for training, 20% for testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 3. Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# 4. Check accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of predictions were correct!')

# 5. Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
    