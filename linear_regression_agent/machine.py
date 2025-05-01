import math

def add_and_log(number1, number2):
    try:
        # Step 2: Add the two numbers
        result = number1 + number2

        # Step 3: Ensure the result is positive for logarithm calculation
        if result <= 0:
            raise ValueError("The sum must be positive to calculate its logarithm.")

        # Step 4: Calculate the logarithm of the result
        log_result = math.log(result)

        # Step 5: Return the result and its logarithm
        return result, log_result

    except ValueError as e:
        return str(e)

# Example usage with predefined values
number1 = 5.0
number2 = 3.0
result, log_result = add_and_log(number1, number2)
print(f"The sum is {result} and the logarithm of the sum is {log_result}")