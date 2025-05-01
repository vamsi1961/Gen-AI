# Function to generate Fibonacci series up to the 10th number
def fibonacci_series(n):
    series = []
    a, b = 0, 1
    while len(series) < n:
        series.append(a)
        a, b = b, a + b
    return series

# Generate Fibonacci series up to the 10th number
fib_series = fibonacci_series(10)

# Print the Fibonacci series
print("Fibonacci series up to 10:", fib_series)

# Print "Hello World"
print("Hello World")