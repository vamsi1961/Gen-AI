def is_sum_greater_than_zero(numbers):
    """
    Check if the sum of the given numbers is greater than zero.

    Parameters:
    numbers (list): A list of numbers to sum.

    Returns:
    bool: True if the sum is greater than zero, False otherwise.
    """
    total_sum = sum(numbers)
    return total_sum > 0