## ABSTRACT:

This project focuses on predictive maintenance using a binary classification predictive model. The goal is to anticipate equipment failures, enabling proactive maintenance and minimizing downtime. The outcome is a web app built with Streamlit, allowing users to input machine metrics. The app predicts the likelihood of equipment failure based on a tuned Decision Tree model. The tool provides a user-friendly interface, enhancing accessibility to predictive maintenance insights and contributing to more efficient and informed decision-making in industrial settings.
 
## DATA DESCRIPTION:
Here's a brief description of each column in the dataset:

**UDI (Unique Device Identifier):** A unique identifier assigned to each device or piece of equipment. It serves as a distinct label for tracking and identification purposes.

**Product ID:** An identifier associated with the specific product or equipment in the dataset. It helps in categorizing and differentiating between various products.

**Type:** Denotes the type of the equipment or device. This categorical variable may represent different classes or categories of equipment.

**Air temperature [K]:** The temperature of the air surrounding the equipment, measured in Kelvin. It provides information about the environmental conditions during the equipment's operation.

**Process temperature [K]:** The temperature of the equipment's internal processes, measured in Kelvin. It indicates the thermal conditions within the equipment during operation.

**Rotational speed [rpm]:** The speed at which the equipment rotates, measured in revolutions per minute (rpm). This parameter reflects the rotational dynamics of the equipment.

**Torque [Nm]:** The amount of force applied to the equipment, measured in Newton-meters (Nm). Torque is a critical factor in understanding the mechanical stress on the equipment.

**Tool wear [min]:** The duration for which the equipment's tool has been in operation, measured in minutes. Tool wear is a key metric for assessing the wear and tear of equipment tools.

**Target:** A binary classification label indicating whether a failure has occurred or not. It serves as the target variable for binary classification tasks.

**Failure Type:** For multiclass classification, this column specifies the type or category of failure when a failure occurs. It serves as the target variable for multiclass classification tasks.

These columns collectively provide comprehensive information about the equipment's operational parameters, environmental conditions, and the occurrence and type of failures.

## ALGORITHM DESCRIPTION:
The main alogorithm powering the machine failure perediction (binary classification) web app is an  optimized Decision Tree classifier. 
### Data preprocessing:
1. Columns such as IDs (UDI, Product ID) and categorical variables (Type, Failure Type) are removed from the dataset. These columns are often dropped if they do not contribute valuable information to the predictive model or if they are identifiers without predictive power.
2. The dataset is divided into two subsets: a training set and a test set. The training set is used to train the machine learning model, and the test set is kept separate to evaluate the model's performance on unseen data. This helps assess how well the model generalizes to new instances.
3. Numerical features like temperatures (Air temperature [K], Process temperature [K]) and torque (Torque [Nm]) are standardized. Standardization involves transforming the data to have a mean of 0 and a standard deviation of 1. This is beneficial for certain machine learning algorithms that are sensitive to the scale of input features, ensuring they contribute equally to the model.
### Model training:
1. A Decision Tree model is a machine learning algorithm that makes decisions by recursively splitting the dataset based on the features. The first step involves training this model on the training data (X_binary_train, y_binary_train). The model learns to map the input features to the target variable, which, in this case, is binary (Target).
2. Hyperparameters are configuration settings for a machine learning algorithm that are not learned from the data but need to be specified beforehand. In this case, hyperparameters like max_depth (maximum depth of the tree) and min_samples (minimum number of samples required to split an internal node) are tuned to find the combination that optimizes the model's performance. GridSearch involves trying out different combinations of hyperparameter values in a specified grid and selecting the one that yields the best performance.
3. The F1-score is a metric that combines precision and recall, providing a balanced measure of a model's performance. The best-performing configuration is determined by selecting the hyperparameter values that result in the highest F1-score. This step ensures that the model performs well in terms of both correctly identifying positive instances (precision) and capturing all positive instances (recall).
### Making predictions:
1. The web app prompts users to input specific machine metrics, such as air temperature, process temperature, rotational speed, torque, and tool wear. These metrics serve as the features that the machine learning model uses for prediction.
2. To ensure that the user input is compatible with the model, the numerical variables in the input data (e.g., temperatures, torque) are standardized. Standardization involves scaling the values so that they have a mean of 0 and a standard deviation of 1. This ensures that the input is consistent with the scale used during the training of the model.
3. The preprocessed user input data is then passed through the tuned Decision Tree model. The model has been previously trained and optimized using the training data. It uses the learned patterns to make predictions on the user's input.
4. Instead of providing a binary prediction directly, the model outputs class probabilities. In binary classification, this typically means providing the probability of belonging to the positive class (e.g., machine failure). The predicted probabilities give an indication of the model's confidence in its prediction.
5. A threshold is set to determine the classification outcome. By default, this threshold is often set to 0.5. If the predicted probability of failure is equal to or exceeds the threshold, the machine is classified as experiencing failure; otherwise, it is classified as working.
### Final app:
1. The web app displays the predicted failure status to the user. This binary outcome informs the user whether the machine is predicted to be in a normal working condition or if there's a likelihood of failure based on the input metrics.
2. Alongside the predicted failure status, the app also shows the calculated failure probability. This is the model's estimation of the likelihood that the machine will experience failure. It gives users insights into the model's confidence regarding the predicted outcome.
3. The web app allows users to interactively modify the probability threshold. By default, a threshold of 0.5 is often used, meaning that if the predicted probability of failure is equal to or exceeds 0.5, the machine is classified as experiencing failure. Users have the flexibility to adjust this threshold based on their preferences or specific operational requirements. This feature enables users to fine-tune the model's sensitivity and specificity according to their needs.

## TOOLS USED:
1. Python: Python is the primary programming language used for data preprocessing, analysis, and building machine learning models. It offers extensive libraries and frameworks suitable for these tasks.
2. Pandas: Pandas is used for data manipulation and analysis. It provides data structures like DataFrames, which are particularly useful for handling and processing structured data.
3. Numpy: NumPy is a fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays.
4. Sci-kit: Scikit-learn is a machine learning library in Python. It includes various tools for classification, regression, clustering, and more. In this project, it's used for building, training, and optimizing machine learning models.
5. Streamlit: Streamlit is employed to create web applications with minimal effort. It allows for the easy conversion of data scripts into shareable web apps. In this project, Streamlit is used to develop the interactive machine failure prediction app.
6. Matplotlib: Matplotlib is used for data visualization. They are utilized to create plots and charts, aiding in the exploratory data analysis (EDA) phase of the project.
7. StandardScaler from Scikit-learn: StandardScaler is a preprocessing tool used to standardize numerical features. It ensures that the features have a mean of 0 and a standard deviation of 1, which is often a requirement for certain machine learning algorithms.
8. DecisionTreeClassifier from Scikit-learn: The DecisionTreeClassifier is a machine learning algorithm used for classification tasks. It's employed to train a model that predicts equipment failure based on input features.
9. GridSearchCV from Scikit-learn: GridSearchCV is a tool for hyperparameter tuning. It performs an exhaustive search over a specified parameter grid to identify the best hyperparameter configuration for a given machine learning model.

## WEB APP URL:
Here is the webapp url for the binary classification machine failure prediction app: https://binaryclass.streamlit.app

## (Optional) Multiclass classification predictive model description:
Apart from binary classification, I've performed multiclass classification as well. Here is the description oft he algorithm used.
### Data preprocessing:
1. **OneHotEncode:** In the multiclass classification task, the target variable is "Failure Type," which contains categorical labels indicating the type of failure. The OneHotEncoder is employed to encode these categorical labels into a binary matrix format. This process is known as one-hot encoding, where each unique category becomes a separate binary column. The resulting binary matrix helps in providing a suitable format for the machine learning algorithms to understand and learn from the categorical labels.
2. **StandardScaler - Standardizing the Numerical Features:** The numerical features in the dataset, such as "Air temperature [K]," "Process temperature [K]," "Rotational speed [rpm]," "Torque [Nm]," and "Tool wear [min]," have varying scales. The StandardScaler is utilized to standardize these numerical features, ensuring that they all have a mean of 0 and a standard deviation of 1. Standardization is important for certain machine learning algorithms, particularly those sensitive to the scale of input features.
### Model training:
1. The RandomForestClassifier is a machine learning algorithm belonging to the ensemble learning category, specifically the Random Forest ensemble. It builds multiple decision trees and merges them together to get a more accurate and stable prediction.
2. The GridSearchCV tool is employed for hyperparameter tuning. It performs an exhaustive search over a specified parameter grid, trying all possible combinations of hyperparameter values. In this case, hyperparameters like 'n_estimators' (number of trees in the forest), 'max_depth' (maximum depth of the tree), 'min_samples_split' (minimum samples required to split an internal node), and 'min_samples_leaf' (minimum samples required to be at a leaf node) are considered.