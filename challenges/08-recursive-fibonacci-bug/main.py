"""Challenge: Fix edit distance (Levenshtein distance) implementation with memoization."""

def edit_distance(s1, s2, memo=None):
    """
    Calculate the minimum edit distance (Levenshtein distance) between two strings.

    The edit distance is the minimum number of single-character edits (insertions,
    deletions, substitutions) required to change one string into another.

    Examples:
    - edit_distance("cat", "cat") = 0 (identical)
    - edit_distance("cat", "dog") = 3 (replace all 3 characters)
    - edit_distance("kitten", "sitting") = 3 (substitute k->s, e->i, insert g)

    Algorithm:
    - If either string is empty, distance is the length of the other string
    - If first characters match, recurse on the rest: edit_distance(s1[1:], s2[1:])
    - If they don't match, take minimum of:
      * Insert: 1 + edit_distance(s1, s2[1:])
      * Delete: 1 + edit_distance(s1[1:], s2)
      * Replace: 1 + edit_distance(s1[1:], s2[1:])

    Args:
        s1: First string
        s2: Second string
        memo: Memoization dictionary (internal use)

    Returns:
        Minimum edit distance
    """
    if memo is None:
        memo = {}

    # Create a cache key from the indices
    key = (len(s1), len(s2))
    if key in memo:
        return memo[key]

    # Base cases
    if len(s1) == 0:
        result = len(s2)
    elif len(s2) == 0:
        result = len(s1)
    # Recursive case: BUG - comparison and recursion have errors
    elif s1[0] == s2[0]:
        # Characters match: don't count as an edit
        result = edit_distance(s1[1:], s2[1:], memo)
    else:
        # Characters don't match: try all three operations
        insert = edit_distance(s1, s2[1:], memo)
        delete = edit_distance(s1[1:], s2, memo)
        replace = edit_distance(s1[1:], s2[1:], memo)
        result = 1 + min(insert, delete)  # BUG: Missing 'replace' in min() call

    memo[key] = result
    return result

def main():
    """Test edit distance with various string pairs."""
    test_cases = [
        ("", ""),
        ("a", ""),
        ("", "b"),
        ("cat", "cat"),
        ("cat", "dog"),
        ("kitten", "sitting"),
        ("saturday", "sunday"),
        ("abc", "def"),
        ("", "hello"),
        ("abc", "abc"),
    ]

    print("Edit distance results:")
    for s1, s2 in test_cases:
        distance = edit_distance(s1, s2)
        print(f'edit_distance("{s1}", "{s2}") = {distance}')

if __name__ == "__main__":
    main()
