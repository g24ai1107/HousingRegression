# regression.py

from utils import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

def grid_search(model, params, X_train, y_train):
    grid = GridSearchCV(model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, r2

def main():
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_params = {
        "Ridge": (Ridge(), {
            "alpha": [0.01, 0.1, 1, 10],
            "fit_intercept": [True, False],
            "solver": ["auto", "saga"]
        }),
        "DecisionTree": (DecisionTreeRegressor(), {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }),
        "SVR": (SVR(), {
            "kernel": ["rbf", "linear"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"]
        }),
    }

    for name, (model, params) in models_params.items():
        print(f"\nTuning {name}...")
        best_model = grid_search(model, params, X_train, y_train)
        mse, r2 = evaluate(best_model, X_test, y_test)
        print(f"{name} => MSE: {mse:.2f}, R²: {r2:.2f}")

if __name__ == "__main__":
    main()

