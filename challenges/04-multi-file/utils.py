def sum_range(start: int, end: int) -> int:
    """
    Return the sum of integers in [start, end].
    """
    total = 0
    for value in range(start, end):
        total += value
    return total