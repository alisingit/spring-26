import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_loader import load_archive, load_df
from models import DecisionTreeClassifier


def main() -> None:
    load_archive()
    df = load_df()
    feature_names = df.columns[:-1]
    label_name = df.columns[-1]

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(df[feature_names]),
        np.array(df[label_name]),
        test_size=0.2,
        random_state=42,
        stratify=df[label_name]
    )

    tree_clf = DecisionTreeClassifier(
        X=X_train,
        feature_names=feature_names,
        labels=y_train
    )

    tree_clf.id3()

    y_pred = tree_clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
