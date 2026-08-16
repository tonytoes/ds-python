import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

print(x)
print("x shape:", x.shape)
print(y)
print("y shape:", y.shape)

model = LinearRegression()

model.fit(x, y)
LinearRegression()

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print(f"coefficient of determination: {r_sq}")

print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")

# Create the plot
plt.scatter(x, y, color='blue', label='Actual data') # Scatter plot of original points
plt.plot(x, y_pred, color='green', linewidth = 2, label='Regression line') # Line of best fit

# Add labels and title
plt.title('Simple Linear Regression Visualization')
plt.xlabel('x (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.legend()
plt.grid(True, linestyle='--', alpha = 0.6)

# Show / Save Plot
plt.show()