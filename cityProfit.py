import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')  # Changed the style to 'ggplot'

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
input_size = np.array([1.0, 2.0])  # Changed x_train to input_size
target_price = np.array([300.0, 500.0])  # Changed y_train to target_price
print(f"input_size = {input_size}")
print(f"target_price = {target_price}")

# m is the number of training examples
print(f"input_size.shape: {input_size.shape}")
num_examples = input_size.shape[0]  # Changed m to num_examples
print(f"Number of training examples is: {num_examples}")

example_index = 0  # Change this to 1 to see (x^1, y^1)

current_input = input_size[example_index]  # Changed x_i to current_input
current_target = target_price[example_index]  # Changed y_i to current_target
print(f"(x^({example_index}), y^({example_index})) = ({current_input}, {current_target})")

# Plot the data points
plt.scatter(input_size, target_price, marker='o', c='b')  # Changed marker to 'o' and color to 'b'
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

initial_weight = 100  # Changed w to initial_weight
initial_bias = 100  # Changed b to initial_bias
print(f"initial_weight: {initial_weight}")
print(f"initial_bias: {initial_bias}")

def calculate_predicted_values(x, weight, bias):
    """
    Calculates the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      weight, bias (scalar): model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    num_examples = x.shape[0]
    predicted_values = np.zeros(num_examples)
    for i in range(num_examples):
        predicted_values[i] = weight * x[i] + bias
        
    return predicted_values

predicted_values = calculate_predicted_values(input_size, initial_weight, initial_bias)

# Plot our model prediction
plt.plot(input_size, predicted_values, c='r', label='Our Prediction')  # Changed color to 'r'

# Plot the data points
plt.scatter(input_size, target_price, marker='x', c='b', label='Actual Values')  # Changed color to 'b'

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

updated_weight = 200  # Changed w to updated_weight
initial_bias = 100  # Kept b as initial_bias
input_value = 1.2  # Changed x_i to input_value
predicted_price = updated_weight * input_value + initial_bias  # Changed cost_1200sqft to predicted_price

print(f"${predicted_price:.0f} thousand dollars")
