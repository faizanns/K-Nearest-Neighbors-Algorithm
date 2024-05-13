import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

class KNN:
    def __init__(self, k = 3, weight = 'uniform'):
        self.k = k
        self.weight = weight

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def euclidian_Distance(self, X_train, test):
        # returns distance
        return np.sqrt(np.sum(np.square(np.subtract(X_train, test))))

    def predict(self, test):
        predictions = []
        for j in range(0, len(test)):
            distances = [self.euclidian_Distance(train_data_row, test[j]) for train_data_row in self.X_train]
            # indices of nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # labels of k neighbors
            k_nearest_labels = [self.Y_train[k_indices[i]] for i in range(0, np.size(k_indices))]
            
            # majority vote        
            if self.weight == 'uniform':
                # non-weighted votes
                # get the most repeating label i.e. np.bincount counts no. of each label, np.argmax outputs index of max count
                prediction = k_nearest_labels[np.argmax(np.bincount(k_nearest_labels))]
            elif self.weight == 'distance':
                # weighted votes
                # weights are the reciprocal of the k distances
                weights = [1 / distances[k_indices[i]] for i in range(0, np.size(k_indices))]
                # weights are divided by sum of all weights to get the votes
                weighted_votes = [weights[i] / np.sum(weights) for i in range(0, len(weights))]
                # get the most repeating weighted vote i.e. np.bincount counts no. of each weighted votes, np.argmax outputs index of max count
                prediction = k_nearest_labels[np.argmax(np.bincount(weighted_votes))]
            predictions.append(prediction)
            
        return predictions

# Code made to work with the following inputs:
# Use this data as your reference data points and labels

# Classification data
cl_X = np.arange(0, 9).reshape(9,1)
cl_y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
print('Classification input:\n', X, '\tShape:', cl_X.shape)
print('Classification labels:\n', y, '\tShape:', cl_y.shape)


# Regression data
r_X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
r_y = np.array([1, 2, 3, 4, 5])
print('Regression input:\n', r_X, '\tShape:', r_X.shape)
print('Regression labels:\n', r_y, '\tShape:', r_y.shape)


# Use the following list to test your code for classification
cl_X_test = np.array([[2.1], [5.2], [7.2]])

r_X_test = np.array([[2.1, 5.1], [2.6, 6.2]])

# classification (dont forget the weighting function)

# Non-weighted
knn_clf = KNN()
knn_clf.fit(cl_X, cl_y)
print(knn_clf.predict(cl_X_test))

# Weighted
knn_clf = KNN(weight='distance')
knn_clf.fit(cl_X, cl_y)
print(knn_clf.predict(cl_X_test))

# sklearn comparison
#Non-weighted
knn_clf_sklearn = KNeighborsClassifier(n_neighbors=3)
knn_clf_sklearn.fit(cl_X, cl_y)
print(knn_clf_sklearn.predict(cl_X_test))

#Weighted
knn_clf_sklearn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn_clf_sklearn.fit(cl_X, cl_y)
print(knn_clf_sklearn.predict(cl_X_test))


# regression (dont forget the weighting function)

# Non-weighted
knn_regressor = KNN()
knn_regressor.fit(r_X, r_y)
print(knn_clf.predict(r_X_test))

# Weighted
knn_regressor = KNN(weight='distance')
knn_regressor.fit(r_X, r_y)
print(knn_clf.predict(r_X_test))

# sklearn comparison
# Non-weighted
knn_regressor_sklearn = KNeighborsRegressor(n_neighbors=3)
knn_regressor_sklearn.fit(r_X, r_y)
print(knn_regressor_sklearn.predict(r_X_test))

# Weighted
knn_regressor_sklearn = KNeighborsRegressor(n_neighbors=3, weights='distance')
knn_regressor_sklearn.fit(r_X, r_y)
print(knn_regressor_sklearn.predict(r_X_test))
