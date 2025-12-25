"""Challenge harness for 04-decimal-rounding."""

from decimal import Decimal, ROUND_HALF_UP


def calculate_invoice(items: list[tuple[str, float]], tax_rate: float) -> tuple[Decimal, Decimal, Decimal]:
    """
    Calculate subtotal, tax, and total using Decimal math.

    Rules:
    1. Convert numeric inputs to Decimal WITHOUT using binary floats.
    2. Tax is subtotal * tax_rate.
    3. Round money to 2 decimal places using ROUND_HALF_UP.
    """
    subtotal = Decimal("0.00")
    for _, price in items:
        subtotal += Decimal(price)
    tax = subtotal * Decimal(tax_rate)
    total = subtotal + tax
    return subtotal, tax, total


def format_money(amount: Decimal) -> str:
    return f"{amount.quantize(Decimal('0.01'))}"


def main() -> None:
    cases = [
        (
            [("hammer", 19.75), ("wrench", 14.50), ("pliers", 10.25)],
            0.05,
            ("44.50", "2.23", "46.73"),
        ),
        (
            [("book", 12.30), ("pen", 0.10), ("notebook", 4.60)],
            0.095,
            ("17.00", "1.62", "18.62"),
        ),
        (
            [("sticker", 0.01), ("clip", 0.03)],
            0.125,
            ("0.04", "0.01", "0.05"),
        ),
    ]

    failures = []
    for index, (items, rate, expected) in enumerate(cases, start=1):
        subtotal, tax, total = calculate_invoice(items, rate)
        actual = (format_money(subtotal), format_money(tax), format_money(total))
        if actual != expected:
            failures.append((index, expected, actual))
        else:
            print(f"case_{index}: PASSED")

    if failures:
        for index, expected, actual in failures:
            print(f"case_{index}: FAILED")
            print(f"expected={expected}")
            print(f"actual={actual}")
        raise SystemExit(1)

    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        raise
