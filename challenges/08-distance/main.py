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
