import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("predictive_maintenance.csv")

# Drop unnecessary columns
df_multi = df.drop(["UDI", "Product ID", "Type"], axis=1, errors="ignore")

# Train-test split
X_multi = df_multi.drop("Failure Type", axis=1)
y_multi = df_multi["Failure Type"]
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X_multi_train_imputed = pd.DataFrame(imputer.fit_transform(X_multi_train), columns=X_multi_train.columns)
X_multi_test_imputed = pd.DataFrame(imputer.transform(X_multi_test), columns=X_multi_test.columns)

# Data cleaning
ohe = OneHotEncoder()
y_multi_train_encoded = ohe.fit_transform(y_multi_train.values.reshape(-1, 1)).toarray()

# Feature scaling
to_scale = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
sc = StandardScaler()
X_multi_train_imputed[to_scale] = sc.fit_transform(X_multi_train_imputed[to_scale])
X_multi_test_imputed[to_scale] = sc.transform(X_multi_test_imputed[to_scale])

# Load the trained model
best_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
best_model.fit(X_multi_train_imputed, y_multi_train_encoded)

# Streamlit app
def main():
    st.title("Machine Failure Prediction App")

    # User input for machine features
    air_temp = st.number_input("Air Temperature [K]", value=298.0, min_value=290.0, max_value=310.0)
    process_temp = st.number_input("Process Temperature [K]", value=310.0, min_value=300.0, max_value=320.0)
    rotational_speed = st.number_input("Rotational Speed [rpm]", value=1500, min_value=1000, max_value=3000)
    torque = st.number_input("Torque [Nm]", value=40.0, min_value=1.0, max_value=80.0)
    tool_wear = st.number_input("Tool Wear [min]", value=5, min_value=0, max_value=260)

    # Predict button
    if st.button("Predict Failure Type"):
        user_input = {"Air temperature [K]": air_temp,
                      "Process temperature [K]": process_temp,
                      "Rotational speed [rpm]": rotational_speed,
                      "Torque [Nm]": torque,
                      "Tool wear [min]": tool_wear}

        # Preprocess the input and make prediction
        input_data = pd.DataFrame([user_input], columns=X_multi_train_imputed.columns)
        input_data[to_scale] = sc.transform(input_data[to_scale])

        # Make prediction
        prediction_probs = best_model.predict_proba(input_data)

        # Find the predicted class
        predicted_class_index = np.argmax(prediction_probs)
        predicted_class = ohe.categories_[0][predicted_class_index]

        # Display the result
        st.subheader("Prediction:")
        st.write(f"The predicted failure type is: {predicted_class}")

if __name__ == "__main__":
    main()



