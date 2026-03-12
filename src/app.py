import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler


DATA_URL = "https://storage.googleapis.com/breathecode/project-files/bank-marketing-campaign-data.csv"


def load_data():
    data = pd.read_csv(DATA_URL, sep=";")
    return data.drop_duplicates()


def prepare_data(data):
    x = data.drop("y", axis=1)
    y = data["y"].map({"no": 0, "yes": 1})

    x = pd.get_dummies(x, drop_first=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    selector = SelectKBest(score_func=chi2, k=5)
    selector.fit(x_train_scaled, y_train)

    selected_mask = selector.get_support()
    selected_features = x_train.columns[selected_mask]

    x_train_sel = pd.DataFrame(x_train_scaled, columns=x_train.columns).loc[:, selected_features]
    x_test_sel = pd.DataFrame(x_test_scaled, columns=x_test.columns).loc[:, selected_features]

    return x_train_sel, x_test_sel, y_train, y_test, selected_features


def evaluate_model(model, x_test, y_test, label):
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    print(f"\n{label}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


def main():
    data = load_data()
    x_train_sel, x_test_sel, y_train, y_test, selected_features = prepare_data(data)

    print("Selected features:")
    print(list(selected_features))

    base_model = LogisticRegression(random_state=42, max_iter=1000)
    base_model.fit(x_train_sel, y_train)
    evaluate_model(base_model, x_test_sel, y_test, "Base model")

    hyperparams = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }

    grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        hyperparams,
        scoring="accuracy",
        cv=5,
    )
    grid.fit(x_train_sel, y_train)

    print("\nBest params:")
    print(grid.best_params_)

    best_model = grid.best_estimator_
    evaluate_model(best_model, x_test_sel, y_test, "Optimized model")


if __name__ == "__main__":
    main()
