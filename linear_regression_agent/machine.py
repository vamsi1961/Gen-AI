
fibonacci_series = [0, 1]
for i in range(2, 10):
    next_number = fibonacci_series[-1] + fibonacci_series[-2]
    fibonacci_series.append(next_number)
print(fibonacci_series)
