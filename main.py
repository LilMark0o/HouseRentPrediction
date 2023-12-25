from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = 'House_Rent_Dataset.csv'


def getData(moreFeatures=False, scale_features=True):
    df = pd.read_csv(file_path)

    featuresTitles = ["BHK", "Size", "FloorNum",
                      "FloorsAvailable", "Furnishing Status", "Bathroom"]
    if moreFeatures:
        featuresTitles.append("Size*BHK*Bathroom")
    features = np.zeros((df.shape[0], len(featuresTitles)))
    target = np.zeros(df.shape[0])

    for index, row in df.iterrows():
        if row["Rent"] > 1000000:
            continue

        features[index][0] = row["BHK"]
        features[index][1] = row["Size"]
        try:
            features[index][2] = int(row["Floor"].split(" ")[0])
        except:
            features[index][2] = 0
        try:
            features[index][3] = int(row["Floor"].split(" ")[-1])
        except:
            features[index][3] = 0
        if row["Furnishing Status"] == "Unfurnished":
            features[index][4] = 0
        elif row["Furnishing Status"] == "Semi-Furnished":
            features[index][4] = 1
        else:
            features[index][4] = 2

        features[index][5] = row["Bathroom"]

        if moreFeatures:
            features[index][6] = row["Size"] * row["BHK"] * row["Bathroom"]

        target[index] = row["Rent"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def trainData(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predictData(model, X_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return predictions, mae, mse, r2


def showData(X_test, y_test, predictions):
    # Visualize predictions vs. actual values
    plt.scatter(X_test[:, 1], y_test, color='black',
                label='Actual')  # 'Size' is at index 1
    plt.scatter(X_test[:, 1], predictions, color='blue', label='Predicted')
    plt.xlabel('Size')
    plt.ylabel('Rent')
    plt.legend()
    plt.show()


def compareMetrics(mae1, mse1, r21, mae2, mse2, r22):
    print("\nMean Absolute Error (MAE):")
    print(f'Mean Absolute Error: {mae1} vs {mae2}')
    print(f"Winner: {'Model 1' if mae1 < mae2 else 'Model 2'}")

    print("\nMean Squared Error (MSE):")
    print(f'Mean Squared Error: {mse1} vs {mse2}')
    print(f"Winner: {'Model 1' if mse1 < mse2 else 'Model 2'}")

    print("\nR-squared (R2):")
    print(f'R-squared: {r21} vs {r22}')
    print(f"Winner: {'Model 1' if r21 > r22 else 'Model 2'}")


def compareModels(predictions1, predictions2, y_test):
    residuals1 = y_test - predictions1

    residuals2 = y_test - predictions2

    plt.scatter(predictions1, residuals1, label='Model 1', color='blue')
    plt.scatter(predictions2, residuals2, label='Model 2', color='red')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.title('Residual Plot')
    plt.show()


def compareModels2(predictions1, predictions2, y_test):
    residuals1 = y_test - predictions1
    residuals2 = y_test - predictions2

    plt.hist(residuals1, bins=50, label='Model 1', color='blue', alpha=0.7)

    plt.hist(residuals2, bins=50, label='Model 2', color='red', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Residuals')
    plt.show()


X_train, X_test, y_train, y_test = getData()
X_train2, X_test2, y_train2, y_test2 = getData(True)
model = trainData(X_train, y_train)
model2 = trainData(X_train2, y_train2)
predictions, mae, mse, r2 = predictData(model, X_test)
predictions2, mae2, mse2, r22 = predictData(model2, X_test2)
compareMetrics(mae, mse, r2, mae2, mse2, r22)
compareModels(predictions, predictions2, y_test)
compareModels2(predictions, predictions2, y_test)
