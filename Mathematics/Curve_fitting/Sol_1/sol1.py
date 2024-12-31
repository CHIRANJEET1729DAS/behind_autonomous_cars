import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic data
torch.manual_seed(42)
x = torch.linspace(-10, 10, steps=100).unsqueeze(1) 
true_coefficients = [2, -3, 1.5] 
y = sum(c * (x ** i) for i, c in enumerate(true_coefficients)) + torch.randn_like(x) * 2  

# Plot the generated data points
plt.scatter(x.numpy(), y.numpy(), label="Generated Data", color='blue')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Generated Data Points")
plt.legend()
plt.show()

# Define the degree of the polynomial
degree = len(true_coefficients) - 1


class PolynomialFitModel(nn.Module):
    def __init__(self, degree):
        super(PolynomialFitModel, self).__init__()
        self.poly = nn.Linear(degree + 1, 1, bias=False)  

    def forward(self, x):
        # Create polynomial terms (x^0, x^1, x^2, ..., x^degree)
        powers = torch.cat([x ** i for i in range(degree + 1)], dim=1)
        return self.poly(powers)

# Create model, define loss function and optimizer
model = PolynomialFitModel(degree)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Prepare the data loader
dataset = torch.utils.data.TensorDataset(x, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# Train the model
epochs = 500
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

# Extract coefficients from the trained model
coefficients = model.poly.weight.detach().numpy().flatten()

# Print the polynomial equation
equation_terms = [f"{coeff:.4f}x^{i}" for i, coeff in enumerate(coefficients)]
equation = " + ".join(equation_terms)
print("\nLearned Polynomial Equation:")
print(f"y = {equation}")

# Plot the fitted curve
x_fit = torch.linspace(-10, 10, steps=100).unsqueeze(1)
y_fit = model(x_fit).detach()

plt.scatter(x.numpy(), y.numpy(), label="Generated Data", color='blue')
plt.plot(x_fit.numpy(), y_fit.numpy(), label="Fitted Curve", color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Fitted Polynomial Curve")
plt.legend()
plt.show()

