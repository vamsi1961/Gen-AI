import math

# Function to add two numbers and calculate the logarithm of the result
def add_and_logarithm(num1, num2):
    result = num1 + num2
    if result > 0:
        log_result = math.log(result)
        return log_result
    else:
        return "Logarithm undefined for non-positive numbers."

# Predefined numbers for testing
number1 = 5.0
number2 = 3.0

# Calculate and print the logarithm of their sum
logarithm_result = add_and_logarithm(number1, number2)
print("The logarithm of the sum is:", logarithm_result)