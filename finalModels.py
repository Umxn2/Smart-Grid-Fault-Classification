import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
# Preprocess data
def preprocess_data(train_data, test_data):
    X_train = train_data[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values
    y_train = train_data['Fault_Type'].values
    X_test = test_data[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values
    y_test = test_data['Fault_Type'].values

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Feature scaling
#    scaler = StandardScaler()
#    X_train = scaler.fit_transform(X_train)
#    X_val = scaler.transform(X_val)
#    X_test = scaler.transform(X_test)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Define ANN model
def create_ann_model(input_shape, output_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,), kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(output_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train ANN
def train_ann_model(X_train, y_train, X_val, y_val, output_classes):
    model = create_ann_model(X_train.shape[1], output_classes)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    checkpoint = ModelCheckpoint('ann_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

    model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    return model

# Train Naive Bayes
def train_naive_bayes(X_train, y_train):
    model = Pipeline([
      #  ('scaler', StandardScaler()),
        ('nb', GaussianNB())
    ])
    model.fit(X_train, y_train)
    dump(model, 'naive_bayes_model.joblib')  # Save model
    return model

# Train Random Forest
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    }
    
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    dump(model, 'random_forest_model.joblib')  # Save model
    return model

# Train SVM
def train_svm(X_train, y_train):
    model = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train, y_train)
    dump(model, 'svm_model.joblib')  # Save model
    return model

# Evaluate Models
def evaluate_model(model, X_test, y_test, is_ann=False):
    if is_ann:
        print("ANN REPORT")
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
# Load datasets
trainDataset = pd.read_csv('./Dataset/train_dataset.csv')
testDataset = pd.read_csv('./Dataset/test_dataset_with_fault_type.csv')

# Preprocess data
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(trainDataset, testDataset)

# Train and Evaluate ANN
ann_model = train_ann_model(X_train, y_train, X_val, y_val, len(np.unique(y_train)))
evaluate_model(ann_model, X_test, y_test, is_ann=True)
# Train and Evaluate Naive Bayes
nb_model = train_naive_bayes(X_train, y_train)
print("NAIVE BAYES REPORT")
evaluate_model(nb_model, X_test, y_test)
# Train and Evaluate Random Forest
rf_model = train_random_forest(X_train, y_train)

print("RANDOM FOREST REPORT")
evaluate_model(rf_model, X_test, y_test)
# Train and Evaluate SVM
svm_model = train_svm(X_train, y_train)

print("SVM REPORT")
evaluate_model(svm_model, X_test, y_test)

