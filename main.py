import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("predictive_maintenance.csv")

# Drop unnecessary columns
df_binary = df.drop(["UDI", "Product ID", "Type", "Failure Type"], axis=1, errors="ignore")

# Train-test split
X_binary = df_binary.drop("Target", axis=1)
y_binary = df_binary["Target"]
X_binary_train, X_binary_test, y_binary_train, y_binary_test = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

# Feature scaling
to_scale = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
sc = StandardScaler()
X_binary_train[to_scale] = sc.fit_transform(X_binary_train[to_scale])
X_binary_test[to_scale] = sc.transform(X_binary_test[to_scale])

# Load the optimized Decision Tree model
best_model = DecisionTreeClassifier(max_depth=6, min_samples_split=2, min_samples_leaf=2, random_state=42)
best_model.fit(X_binary_train, y_binary_train)

# Streamlit app
def main():
    st.title("Machine Failure Prediction App (Binary Classification)")

    # User input for machine features
    air_temp = st.number_input("Air Temperature [K]", value=298.0, min_value=290.0, max_value=310.0)
    process_temp = st.number_input("Process Temperature [K]", value=310.0, min_value=300.0, max_value=320.0)
    rotational_speed = st.number_input("Rotational Speed [rpm]", value=1500, min_value=1000, max_value=3000)
    torque = st.number_input("Torque [Nm]", value=40.0, min_value=1.0, max_value=80.0)
    tool_wear = st.number_input("Tool Wear [min]", value=5, min_value=0, max_value=260)

    # Probability threshold slider
    threshold = st.slider("Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Predict button
    if st.button("Predict Machine Failure"):
        user_input = {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }

        # Preprocess the input and make prediction
        input_data = pd.DataFrame([user_input])
        input_data[to_scale] = sc.transform(input_data[to_scale])

        # Get the probability of the positive class (class 1)
        probability = best_model.predict_proba(input_data)[:, 1]

        # Make prediction based on the probability threshold
        prediction = 1 if probability >= threshold else 0

        # Display the result
        st.subheader("Prediction:")
        st.write(f"The predicted machine failure status is: {'Failure' if prediction == 1 else 'No Failure'}")
        st.write(f"Probability of Failure: {probability[0]:.4f}")

if __name__ == "__main__":
    main()
