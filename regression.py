# regression.py

from utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

def main():
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(),
        "SVR": SVR()
    }

    for name, model in models.items():
        mse, r2 = train_and_evaluate(model, X_train, X_test, y_train, y_test)
        print(f"{name} => MSE: {mse:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    main()

