"""Challenge harness for recursive fibonacci bug fix."""

def fibonacci(n):
    """
    Calculate the nth Fibonacci number using recursion.

    The Fibonacci sequence is defined as:
    - fib(0) = 0
    - fib(1) = 1
    - fib(n) = fib(n-1) + fib(n-2) for n > 1

    Args:
        n: Non-negative integer index

    Returns:
        The nth Fibonacci number
    """
    # BUG: Base case is incorrect
    if n <= 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

def main():
    """Test the fibonacci function with several values."""
    test_cases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print("Fibonacci sequence:")
    for i in test_cases:
        result = fibonacci(i)
        print(f"fib({i}) = {result}")

if __name__ == "__main__":
    main()
