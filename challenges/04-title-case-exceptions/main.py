'''Challenge harness for 03-title-case-exceptions.'''

MINOR_WORDS = {
    "a", "an", "the", "and", "or", "but", "for", "nor",
    "on", "at", "to", "from", "by", "of", "in", "with",
}

def title_case(sentence: str) -> str:
    """Convert a sentence to title case with specified rules."""
    words = sentence.strip().split()
    processed = []
    for i, word in enumerate(words):
        if word.isupper() and len(word) >= 2:
            processed.append(word)
            continue
        if "-" in word:
            subwords = word.split("-")
            processed_subwords = []
            for j, subword in enumerate(subwords):
                is_first_sub = j == 0
                is_last_sub = j == len(subwords) - 1
                is_parent_first = i == 0
                is_parent_last = i == len(words) - 1
                is_first = is_first_sub and is_parent_first
                is_last = is_last_sub and is_parent_last
                if is_first or is_last:
                    processed_subwords.append(subword.capitalize())
                else:
                    if subword.lower() in MINOR_WORDS:
                        processed_subwords.append(subword.lower())
                    else:
                        processed_subwords.append(subword.capitalize())
            processed_word = "-".join(processed_subwords)
        else:
            if i == 0 or i == len(words) - 1:
                processed_word = word.capitalize()
            else:
                if word.lower() in MINOR_WORDS:
                    processed_word = word.lower()
                else:
                    processed_word = word.capitalize()
        processed.append(processed_word)
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