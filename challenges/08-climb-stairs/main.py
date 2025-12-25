"""Challenge harness for climbing stairs dynamic programming bug fix."""

def count_ways_to_climb(n, steps):
    """
    Count the number of distinct ways to climb n steps.

    You can climb 1, 2, or any of the allowed step sizes at a time.
    For example, to reach step 4, you could:
    - Take four 1-steps: [1,1,1,1]
    - Take two 1-steps and one 2-step: [1,1,2], [1,2,1], [2,1,1]
    - Take two 2-steps: [2,2]

    This requires solving the problem: ways(n) = sum of ways(n-step) for each allowed step

    Args:
        n: Total number of steps to climb
        steps: Tuple of allowed step sizes (e.g., (1, 2) means you can take 1 or 2 steps)

    Returns:
        Number of distinct ways to climb exactly n steps
    """
    # BUG: Base case logic is incorrect - should handle 0 and negative cases
    if n < 0:
        return 0
    if n == 0:
        return 0  # BUG: Should be 1 (one way to stay at ground level)

    # Recursive case: sum all ways from previous step sizes
    total = 0
    for step in steps:
        total += count_ways_to_climb(n - step, steps)
    return total

def main():
    """Test the stair climbing function with different step sizes."""
    test_cases = [
        (1, (1,)),      # Can only take 1-steps
        (2, (1,)),      # Can only take 1-steps
        (3, (1, 2)),    # Can take 1 or 2-steps
        (4, (1, 2)),    # Can take 1 or 2-steps
        (5, (1, 2)),    # Can take 1 or 2-steps
        (5, (1, 3)),    # Can take 1 or 3-steps
    ]

    print("Ways to climb stairs:")
    for n, allowed_steps in test_cases:
        result = count_ways_to_climb(n, allowed_steps)
        print(f"climb({n}, {allowed_steps}) = {result}")

if __name__ == "__main__":
    main()
