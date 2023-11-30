import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd. read_csv("predictive_maintenance.csv")

df = df.drop("UDI", axis=1, errors="ignore")
df = df.drop("Product ID", axis=1, errors="ignore")
df = df.drop("Type", axis=1, errors="ignore")

df_multi = df.drop("Target", axis=1, errors="ignore")

X_multi = df_multi.drop("Failure Type", axis=1)
y_multi = df_multi["Failure Type"]

X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)


# Load the trained model
best_model = RandomForestClassifier(random_state=42, max_depth=8, min_samples_split=3, min_samples_leaf=3)
best_model.fit(X_multi_train, y_multi_train)


to_scale = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

sc = StandardScaler()
X_multi_train[to_scale] = sc.fit_transform(X_multi_train[to_scale])
X_multi_test[to_scale] = sc.transform(X_multi_test[to_scale])

# Function to preprocess user input
def preprocess_input(features):
    # Create a DataFrame from the user input
    input_data = pd.DataFrame([features], columns=X_multi_train.columns)

    # Scale the input features
    input_data[to_scale] = sc.transform(input_data[to_scale])

    return input_data

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
        input_data = preprocess_input(user_input)
        prediction = best_model.predict(input_data)

        # Display the result
        st.subheader("Prediction:")
        st.write(f"The predicted failure type is: {prediction[0]}")

if __name__ == "__main__":
    main()
