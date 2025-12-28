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
    # Step 1: trim and collapse whitespace
    words = sentence.strip().split()
    if not words:
        return ""

    def process_word(word: str, is_first: bool, is_last: bool) -> str:
        # Preserve all-caps words (length >=2)
        if word.isupper() and len(word) >= 2:
            return word
        # Handle hyphenated words
        if "-" in word:
            parts = word.split("-")
            new_parts = []
            for idx, part in enumerate(parts):
                part_is_first = is_first and idx == 0
                part_is_last = is_last and idx == len(parts) - 1
                new_parts.append(process_word(part, part_is_first, part_is_last))
            return "-".join(new_parts)
        # Minor word handling
        lowered = word.lower()
        if lowered in MINOR_WORDS and not (is_first or is_last):
            return lowered
        # Capitalize first letter, rest lower
        return word.capitalize()

    processed = []
    last_index = len(words) - 1
    for i, w in enumerate(words):
        processed.append(process_word(w, i == 0, i == last_index))
    return " ".join(processed)

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