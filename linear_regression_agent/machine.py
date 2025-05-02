import math

def add_and_log(num1, num2):
    result = num1 + num2
    if result > 0:
        return math.log(result)
    else:
        return "Logarithm undefined for non-positive numbers"

# Simulating user input for testing
number1 = 5.0  # Example input
number2 = 3.0  # Example input

# Step 2: Use the add_and_log function
log_result = add_and_log(number1, number2)

# Step 3: Print the result
print(f"The logarithm of the sum of {number1} and {number2} is: {log_result}")