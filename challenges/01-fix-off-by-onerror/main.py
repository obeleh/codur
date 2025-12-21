"""Challenge harness for 01-fix-off-by-onerror."""


def sum_range(start: int, end: int) -> int:
    """
    Return the sum of integers in [start, end].
    """
    total = 0
    for value in range(start, end):
        total += value
    return total


def main():
    print(f"sum_range(1, 5) == {sum_range(1, 5)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
