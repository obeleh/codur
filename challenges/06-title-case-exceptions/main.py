'''Challenge harness for 03-title-case-exceptions.'''

MINOR_WORDS = {
    "a", "an", "the", "and", "or", "but", "for", "nor",
    "on", "at", "to", "from", "by", "of", "in", "with",
}

def title_case(sentence: str) -> str:
        """
    Convert a sentence to title case with these rules:

    1. Trim leading/trailing whitespace and collapse internal whitespace.
    2. Split the sentence into words by whitespace.
    3. Preserve words that are already all-caps (length >= 2).
    4. For hyphenated words, split them by hyphen, apply capitalization rules to each subword, and rejoin with hyphens.
       The first subword is considered the "First Word" only if the parent word is the First Word.
       The last subword is considered the "Last Word" only if the parent word is the Last Word.
    5. Capitalize the first and last word of the sentence.
    6. Lowercase minor words unless they are the first or last word.
    7. Capitalize all other words.
    8. Join the processed words back into a sentence with single spaces.
    """
    # TODO: implement
    raise NotImplementedError

def _run_tests() -> None:
    cases = [
        ("the lord of the rings", "The Lord of the Rings"),
        ("a tale of two cities", "A Tale of Two Cities"),
        ("  the rise of NASA and the GPU  ", "The Rise of NASA and the GPU"),
        ("state-of-the-art design", "State-of-the-Art Design"),
        ("from here to eternity", "From Here to Eternity"),
        ("in the middle of nowhere", "In the Middle of Nowhere"),
    ]

    failures = []
    for index, (raw, expected) in enumerate(cases, start=1):
        actual = title_case(raw)
        if actual != expected:
            failures.append((index, raw, expected, actual))
        else:
            print(f"case_{index}: PASSED")

    if failures:
        for index, raw, expected, actual in failures:
            print(f"case_{index}: FAILED")
            print(f"input={raw!r}")
            print(f"expected={expected!r}")
            print(f"actual={actual!r}")
        raise SystemExit(1)

    print("ALL TESTS PASSED!")

def main() -> None:
    _run_tests()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        raise