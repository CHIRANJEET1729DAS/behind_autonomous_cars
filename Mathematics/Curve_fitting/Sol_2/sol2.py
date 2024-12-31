import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
x = np.linspace(-10, 10, 100).reshape(-1, 1)  
true_coefficients = [2, -3, 1.5]  
y = sum(c * (x ** i) for i, c in enumerate(true_coefficients)) + np.random.normal(0, 2, size=x.shape) 

# Plot the generated data points
plt.scatter(x, y, label="Generated Data", color='blue')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Generated Data Points")
plt.legend()
plt.show()

# Degree of the polynomial
degree = len(true_coefficients) - 1

# Polynomial Features Transformation
poly = PolynomialFeatures(degree=degree)
x_poly = poly.fit_transform(x)

# Fit Polynomial Regression Model
model = LinearRegression()
model.fit(x_poly, y)

# Get coefficients and intercept
coefficients = model.coef_.flatten()  # Flatten in case of multi-dimensional arrays
intercept = model.intercept_.item()  # Convert to scalar

# Print the polynomial equation
equation_terms = [f"{coeff:.4f}x^{i}" for i, coeff in enumerate(coefficients)]
equation = f"y = {intercept:.4f} + " + " + ".join(equation_terms[1:])
print("\nLearned Polynomial Equation:")
print(equation)

# Make predictions
y_pred = model.predict(x_poly)

# Calculate and display the mean squared error
mse = mean_squared_error(y, y_pred)
print(f"\nMean Squared Error: {mse:.4f}")

# Plot the fitted curve
plt.scatter(x, y, label="Generated Data", color='blue')
plt.plot(x, y_pred, label="Fitted Curve", color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Fitted Polynomial Curve")
plt.legend()
plt.show()
