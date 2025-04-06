# ABDULRAFAY-197_DEEPLEARNING_LAB02

# ABDULRAFAY(197)_DEEPLEARNING_LAB02

#CODES:
#TASK 01:
#ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 01:
print("ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 01 / STEP FUNCTION:\n")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Python step function
def step_func(t):
    return 1.0 if t >= 0 else 0.0
# Using step function with basic inputs
inputs = [-3.0, -1.0, 0.0, 2.0, 5.0]
outputs = [step_func(t) for t in inputs]
print("Step function output is given as:", outputs)
# TensorFlow-based step function
def tf_step_func(t):
    return tf.cast(tf.math.greater_equal(t, 0), tf.float32)
# Create TensorFlow constant
t = tf.constant([-3.0, -1.0, 0.0, 2.0, 5.0])
print("TensorFlow step output:", tf_step_func(t).numpy())
# Generate values for plotting
x_variable = np.linspace(-5, 5, 100)
y_variable = [step_func(val) for val in x_variable]
# Plotting
plt.plot(x_variable, y_variable)
plt.title("Step Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

#TASK 02:
#ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 02:
print("ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 02 / ReLU FUNCTION:\n")
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Implementing ReLU function manually: f(x) = max(0, x)
def relu_funct(t):
    return max(0.0, t)
# Test ReLU function with specific inputs
inputs = [-4.0, -2.0, 0.0, 3.0, 5.0]
outputs = [relu_funct(t) for t in inputs]
print("ReLU function output for inputs [-4.0, -2.0, 0.0, 3.0, 5.0]:", outputs)
# Expected Output: [0, 0, 0, 3, 5]
# Using TensorFlow ReLU function
def tf_relu(t):
    return tf.nn.relu(t)
# Create TensorFlow tensor with input values
t_tensor = tf.constant([-4.0, -2.0, 0.0, 3.0, 5.0], dtype=tf.float32)
print("TensorFlow ReLU output:", tf_relu(t_tensor).numpy())
# Generate values for plotting
x_variable = np.linspace(-5, 5, 100)  # Input values from -5 to 5 for smooth visualization
x_tensor = tf.constant(x_variable, dtype=tf.float32)  # Convert to tensor for TensorFlow operations
# Apply ReLU to the generated values
y_variable = tf_relu(x_tensor).numpy()
# Plotting the ReLU curve
plt.plot(x_variable, y_variable)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

#TASK 03:
#ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 03:
print("ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 03 / SIGMOID FUNCTION:\n")
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Pure Python Sigmoid function: f(x) = 1 / (1 + e^(-x))
def sigmoid_func(t):
    return 1 / (1 + math.exp(-t))
# Test Sigmoid with basic inputs
inputs = [-2.0, 0.0, 1.0, 3.0]
outputs = [sigmoid_func(t) for t in inputs]
print("Sigmoid function output for inputs [-2.0, 0.0, 1.0, 3.0]:", outputs)
# Expected Output: [0.119, 0.5, 0.731, 0.953]
# TensorFlow Sigmoid function
def tf_sigmoid_func(t):
    return tf.nn.sigmoid(t)
# TensorFlow tensor input
t_tensor = tf.constant([-2.0, 0.0, 1.0, 3.0], dtype=tf.float32)
print("TensorFlow Sigmoid output:", tf_sigmoid_func(t_tensor).numpy())
# Generate values for smooth plotting
x_variable = np.linspace(-5, 5, 100)  # Range from -5 to 5 for a smooth curve
x_tensor = tf.constant(x_variable, dtype=tf.float32)  # Convert to tensor for TensorFlow operation
# Apply Sigmoid to the generated values
y_variable = tf_sigmoid_func(x_tensor).numpy()
# Plotting the Sigmoid curve
plt.plot(x_variable, y_variable)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

#TASK 04:
#ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 04:
print("ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 04 / TanH FUNCTION:\n")
import math
import numpy as np
import matplotlib.pyplot as plt
# Implement Tanh function manually: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
def tanh_manual(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
# Input values
inputs = [-1.5, 0.0, 1.0, 2.0]
# Compute Tanh values for the given inputs
tanh_values = [tanh_manual(x) for x in inputs]
print("Manual Tanh values for inputs [-1.5, 0.0, 1.0, 2.0]:", tanh_values)
# Compare to expected output (approximately): [-0.905, 0.0, 0.762, 0.964]
# Sigmoid function for comparison
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
# Generate values for smooth plotting
x = np.linspace(-5, 5, 100)
# Apply Tanh and Sigmoid to the range of x
y_tanh = np.array([tanh_manual(xi) for xi in x])
y_sigmoid = 1 / (1 + np.exp(-x))
# Plot both Tanh and Sigmoid
plt.plot(x, y_tanh, label="Tanh", color='b')
plt.plot(x, y_sigmoid, label="Sigmoid", color='r', linestyle='--')
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.title("Tanh vs Sigmoid Activation Functions")
plt.legend()
plt.show()

#TASK 05:
#ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 04:
print("ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 04 / LEAKY ReLU FUNCTION:\n")
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Python version of Leaky ReLU: f(x) = x if x >= 0 else alpha * x
def leaky_relu_func(t, alpha=0.01):
    return t if t >= 0.0 else alpha * t
# Example output using Python function
print("Leaky ReLU Value (-2) with alpha=0.1:", leaky_relu_func(-2, 0.1))
# TensorFlow Leaky ReLU function
def tf_leaky_relu_func(t, alpha=0.1):
    return tf.nn.leaky_relu(t, alpha=alpha)
# Tensor input values
t = tf.constant([-3.0, -1.0, 0.0, 2.0])
print("TensorFlow Leaky ReLU Values (alpha=0.1):", tf_leaky_relu_func(t, 0.1).numpy())
# Generate input values for plotting
x_variable = np.linspace(-5, 5, 100)
x_tensor = tf.constant(x_variable, dtype=tf.float32)
# Apply Leaky ReLU for different alpha values
y_alpha_01 = tf_leaky_relu_func(x_tensor, alpha=0.1).numpy()
y_alpha_03 = tf_leaky_relu_func(x_tensor, alpha=0.3).numpy()
# Plotting both on the same graph
plt.plot(x_variable, y_alpha_01, label="alpha = 0.1")
plt.plot(x_variable, y_alpha_03, label="alpha = 0.3", linestyle='--')
plt.grid(True)
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Leaky ReLU Activation Function")
plt.legend()
plt.show()

#TASK 06:
#ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 06:
print("ABDUL RAFAY / 2022F-BSE-197 / LAB 02 / TASK 06 / SOFTMAX FUNCTION:\n")
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Softmax function implementation: Convert inputs into probabilities
def softmax_func(t):
    ex = np.exp(t - np.max(t))  # To improve numerical stability
    return ex / ex.sum()
# Test Softmax function with input values [1.0, 2.0, 3.0]
t = np.array([1.0, 2.0, 3.0])
print("Softmax Activation Function Output: ", softmax_func(t))
# Expected Output: [0.090, 0.245, 0.665]
# TensorFlow Softmax function
def tf_softmax_func(t):
    return tf.nn.softmax(t)
# TensorFlow tensor input
t_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
print("TensorFlow Softmax Output:", tf_softmax_func(t_tensor).numpy())
# Generate input values for plotting
x_variable = np.linspace(-5, 5, 100)  # Range of values from -5 to 5
x_tensor = tf.constant(x_variable, dtype=tf.float32)  # Convert to tensor for TensorFlow
# Apply Softmax to the generated values
y_variable = tf_softmax_func(x_tensor).numpy()
# Plotting the Softmax curve
plt.plot(x_variable, y_variable)
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.title("Softmax Activation Function")
plt.show()
