# This code takes two numbers as input from the user, adds them together, and stores the result.

def add_two_numbers():
    try:
        # Take two numbers as input from the user
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))
        
        # Add the two numbers
        result = num1 + num2
        
        # Return the result
        return result
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None