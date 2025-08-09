import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

best_knn_regressor = grid_search.best_estimator_
y_pred = best_knn_regressor.predict(X_test)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

X_plot = np.linspace(X_scaled.min(), X_scaled.max(), 500).reshape(-1, 1)
y_plot = best_knn_regressor.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled, y, color='blue', alpha=0.5, label='Data Points')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='KNN Regression Fit')
plt.xlabel("Scaled Feature")
plt.ylabel("Target Value")
plt.title("Advanced K-Nearest Neighbors Regression")
plt.legend()
plt.show()