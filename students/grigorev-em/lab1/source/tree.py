import numpy as np

class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, value=None, left_prob=0.5, right_prob=0.5
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.left_prob = left_prob
        self.right_prob = right_prob

    def is_leaf(self):
        return not self.value is None


class Tree:
    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def gini(self, y: np.ndarray):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    def gini_metrics(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        return (
            (len(y_left) / n) * self.gini(y_left) + (len(y_right) / n) * self.gini(y_right)
        )

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        mask = ~np.isnan(y)
        y_ = y[mask]
        X_ = X[mask, :]

        if len(y_) == 0:
            return Node(value=0)

        if len(np.unique(y_)) == 1:
            return Node(value=int(y_[0]))

        if depth >= self.max_depth or len(y_) < self.min_samples_split:
            return Node(value=self._majority(y_))

        feature, threshold, left_prob, right_prob = self._best_split(X_, y_)
        if feature is None:
            return Node(value=self._majority(y_))

        col = X_[:, feature]
        mask_left = col <= threshold
        mask_right = col > threshold

        left_child = self._build(X_[mask_left], y_[mask_left], depth + 1)
        right_child = self._build(X_[mask_right], y_[mask_right], depth + 1)

        return Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
            left_prob=left_prob,
            right_prob=right_prob
        )

    def _best_split(self, X, y):

        best_gini = float("inf")
        best_feature = None
        best_threshold = None
        best_left_prob = 0.5
        best_right_prob = 0.5

        n_features = X.shape[1]

        for f in range(n_features):
            col = X[:, f]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue
            thresholds = np.unique(valid)
            for t in thresholds:
                mask_left = col <= t
                mask_right = col > t
                y_left = y[mask_left]
                y_right = y[mask_right]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gini = self.gini_metrics(y_left, y_right)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = f
                    best_threshold = float(t)
                    total = mask_left.sum() + mask_right.sum()
                    best_left_prob = mask_left.sum() / total
                    best_right_prob = 1 - best_left_prob

        return best_feature, best_threshold, best_left_prob, best_right_prob

    def predict(self, X):
        
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.is_leaf():
            return int(node.value)

        value = x[node.feature]
        if np.isnan(value):
            if node.left_prob >= node.right_prob:
                return self._predict_one(x, node.left)
            return self._predict_one(x, node.right)

        if value <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def _majority(self, y):
        y_ = y[~np.isnan(y)]
        if len(y_) == 0:
            return 0
        values, counts = np.unique(y_, return_counts=True)
        return values[np.argmax(counts)]

    def prune(self, X_val, y_val):
        self._prune(self.root, X_val, y_val)

    def _prune(self, node, X_val, y_val):
        if node is None or node.is_leaf():
            return
        if node.left:
            self._prune(node.left, X_val, y_val)
        if node.right:
            self._prune(node.right, X_val, y_val)
        if node.left and node.right:
            if node.left.is_leaf() and node.right.is_leaf():
                pred_before = self.predict(X_val)
                acc_before = (pred_before == y_val).mean()

                left_b = node.left
                right_b = node.right
                node.left = None
                node.right = None
                
                node.value = left_b.value
                pred_after = self.predict(X_val)
                acc_after = (pred_after == y_val).mean()
                
                if acc_after < acc_before:
                    node.left = left_b
                    node.right = right_b
                    node.value = None
