from scipy.stats import mode
from enum import global_flag_repr
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model
#Fault_Type	A	B	C	G
#0	1.0	1.0	1.0	0.0
#1	1.0	1.0	1.0	1.0
#2	1.0	1.0	0.0	1.0
#3	1.0	0.0	0.0	1.0
#4	0.0	1.0	1.0	0.0
#5	0.0	0.0	0.0	0.0
fault_type_mapping = {
    0: {'A': 1, 'B': 1, 'C': 1, 'G': 0},
    1: {'A': 1, 'B': 1, 'C': 1, 'G': 1},
    2: {'A': 1, 'B': 1, 'C': 0, 'G': 1},
    3: {'A': 1, 'B': 0, 'C': 0, 'G': 1},
    4: {'A': 0, 'B': 1, 'C': 1, 'G': 0},
    5: {'A': 0, 'B': 0, 'C': 0, 'G': 0},
}
# Display available models
print("Please select a model by entering a number from 1 to 4:")
print("1. Naive Bayes Model")
print("2. Random Forest Model")
print("3. Support Vector Machine (SVM) Model")
print("4. Artificial Neural Network (ANN) Model")

# User input
number: int = int(input())
global_features = None
def request_input():
    print("\nPlease provide three voltage values and three current values:")
    Ia = float(input("Enter Current 1: "))
    Ib = float(input("Enter Current 2: "))
    Ic = float(input("Enter Current 3: "))
    Va = float(input("Enter Voltage 1: "))
    Vb = float(input("Enter Voltage 2: "))
    Vc = float(input("Enter Voltage 3: "))
    features = np.array([[Ia, Ib, Ic, Va, Vb, Vc]])
    global global_features
    global_features = features
    return features

def make_prediction(model):
    features = request_input()
    predictions = model.predict(features)
    print(f"\nPredictions: {predictions}")
    if predictions[0] in fault_type_mapping:
        values = fault_type_mapping[predictions[0]]
        print(f"For Fault_Type {predictions}: A={values['A']}, B={values['B']}, C={values['C']}, G={values['G']}")
    else:
        print("Invalid Fault_Type")

def make_prediction_ann(model):
    features = request_input()
    predictions = model.predict(features)
    predicted_classes = np.argmax(predictions, axis=1)
    print(f"\nPredicted Class: {predicted_classes}")
    if predicted_classes[0] in fault_type_mapping:
        values = fault_type_mapping[predicted_classes[0]]
        print(f"For Fault_Type {predicted_classes}: A={values['A']}, B={values['B']}, C={values['C']}, G={values['G']}")
    else:
        print("Invalid Fault_Type")

# Load and use the selected model
if number == 1:
    print("\nLoading the Naive Bayes model...")
    model = load('./naive_bayes_model.joblib')
    make_prediction(model)
elif number == 2:
    print("\nLoading the Random Forest model...")
    model = load('./random_forest_model.joblib')
    make_prediction(model)
elif number == 3:
    print("\nLoading the Support Vector Machine (SVM) model...")
    model = load('./svm_model.joblib')
    make_prediction(model)
elif number == 4:
    print("\nLoading the Artificial Neural Network (ANN) model...")
    model = load_model('./ann_model.keras')
    make_prediction_ann(model)
else:
    print("\nInvalid input. Please enter a number between 1 and 4.")
print("combined statistics from all models")
ann_model = load_model('./ann_model.keras')
nb_model = load('./naive_bayes_model.joblib')
rf_model = load('./random_forest_model.joblib')
svm_model = load('./svm_model.joblib')
ann_proba = ann_model.predict(global_features)  # ANN probabilities
nb_proba = nb_model.predict_proba(global_features)  # Naive Bayes probabilities
rf_proba = rf_model.predict_proba(global_features)  # Random Forest probabilities
svm_proba = svm_model.predict_proba(global_features)  # SVM probabilities
print(f"Ann: {ann_proba}")
print(f"svm: {svm_proba}")
print(f"nb: {nb_proba}")
print(f"rf: {rf_proba}")
ann_weight = 1.45
rf_weight = 1.55
nb_weight = 1.3
svm_weight = 1.2
combined_proba = (
    ann_weight * ann_proba +
    rf_weight * rf_proba +
    nb_weight * nb_proba +
    svm_weight * svm_proba
)
modelProbabilities = [ann_proba, rf_proba, nb_proba, svm_proba]
models = ["ANN","RF", "NB", "SVM"]
for i in range(0,4):
    print(f"{models[i]} probability is {np.argmax(modelProbabilities[i], axis =1)}")
combined_pred = np.argmax(combined_proba, axis=1)
print(f"Final classification {combined_pred[0]}")
if combined_pred[0] in fault_type_mapping:
    values = fault_type_mapping[combined_pred[0]]
    print(f"For Fault_Type {combined_pred[0]}: A={values['A']}, B={values['B']}, C={values['C']}, G={values['G']}")
else:
    print("Invalid Fault_Type")
