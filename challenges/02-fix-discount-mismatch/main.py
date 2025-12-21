"""Challenge harness for 02-fix-discount-mismatch."""


def calculate_subtotal(items: list[tuple[str, float]]) -> float:
    """Return the subtotal for a list of (name, price) items."""
    total = 0.0
    for _, price in items:
        if price <= 2.0:
            continue
        total += price
    return total


def apply_discount(subtotal: float, code: str) -> float:
    """Apply a percentage discount based on a coupon code."""
    discounts = {
        "SAVE10": 0.10,
        "SAVE20": 0.20,
    }
    rate = discounts.get(code, 0.0)
    return subtotal * (1 - (rate / 100))


def main() -> None:
    items = [
        ("book", 12.50),
        ("pen", 1.20),
        ("notebook", 3.30),
    ]
    subtotal = calculate_subtotal(items)
    discounted = apply_discount(subtotal, "SAVE10")
    print(f"subtotal={subtotal:.2f}")
    print(f"discounted={discounted:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
