import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # 分割特征的索引
        self.threshold = threshold          # 分割阈值
        self.left = left                    # 左子节点
        self.right = right                  # 右子节点
        self.value = value                  # 叶子节点的预测值

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth          # 最大深度
        self.min_samples_split = min_samples_split  # 最小分裂样本数
        self.criterion = criterion          # 分割标准：gini, entropy, mse
        self.root = None                     # 根节点

    def fit(self, X, y):
        self.root = self._grow_tree(np.array(X), np.array(y))

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # 判断任务类型：分类或回归
        is_classification = self.criterion in ['gini', 'entropy']

        # 停止条件
        if (n_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth)):
            return Node(value=self._compute_leaf_value(y))

        if is_classification:
            unique_classes = np.unique(y)
            if len(unique_classes) == 1:
                return Node(value=unique_classes[0])
        else:
            if np.var(y) < 1e-6:
                return Node(value=np.mean(y))

        # 寻找最佳分割
        best_gain = -np.inf
        best_feature, best_threshold = None, None

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_y = y[sorted_indices]

            thresholds = []
            for i in range(1, len(sorted_values)):
                if sorted_values[i] != sorted_values[i-1]:
                    thresholds.append((sorted_values[i] + sorted_values[i-1]) / 2)

            for threshold in thresholds:
                left_indices = feature_values <= threshold
                right_indices = ~left_indices
                left_y = y[left_indices]
                right_y = y[right_indices]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gain = self._compute_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # 如果没有有效分割，返回叶子节点
        if best_gain == -np.inf:
            return Node(value=self._compute_leaf_value(y))

        # 递归构建子树
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature_index=best_feature, threshold=best_threshold,
                    left=left_subtree, right=right_subtree)

    def _compute_leaf_value(self, y):
        if self.criterion in ['gini', 'entropy']:
            classes, counts = np.unique(y, return_counts=True)
            return classes[np.argmax(counts)]
        else:
            return np.mean(y)

    def _compute_gain(self, parent, left, right):
        if self.criterion == 'gini':
            return self._gini_gain(parent, left, right)
        elif self.criterion == 'entropy':
            return self._information_gain(parent, left, right)
        else:
            return self._variance_reduction(parent, left, right)

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _gini_gain(self, parent, left, right):
        gini_p = self._gini(parent)
        n = len(parent)
        gini_l = self._gini(left)
        gini_r = self._gini(right)
        return gini_p - (len(left)/n * gini_l + len(right)/n * gini_r)

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # 防止log(0)

    def _information_gain(self, parent, left, right):
        entropy_p = self._entropy(parent)
        n = len(parent)
        entropy_l = self._entropy(left)
        entropy_r = self._entropy(right)
        return entropy_p - (len(left)/n * entropy_l + len(right)/n * entropy_r)

    def _variance_reduction(self, parent, left, right):
        var_p = np.var(parent)
        n = len(parent)
        var_l = np.var(left)
        var_r = np.var(right)
        return var_p - (len(left)/n * var_l + len(right)/n * var_r)

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in np.array(X)])

    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)
        
if __name__ == "__main__":
    data = load_iris()
    X,y = data.data, data.target 
    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify= y
    )

    model = DecisionTree(criterion='gini')
    model.fit(X_train, y_train)
    acc = np.mean(model.predict(X_test) == y_test)
    print(np.round(acc,3))

