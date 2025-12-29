"""Challenge harness for 02-fix-discount-mismatch."""


def calculate_subtotal(items: list[tuple[str, float]]) -> float:
    """Return the subtotal for a list of (name, price) items."""
    total = 0.0
    for _, price in items:
        total += price
    return total


def apply_discount(subtotal: float, code: str) -> float:
    """Apply a percentage discount based on a coupon code.

    The discounts dictionary stores the discount rate as a decimal fraction (e.g., 0.10 for 10%).
    The function returns the subtotal after applying the discount.
    """
    discounts = {
        "SAVE10": 0.10,
        "SAVE20": 0.20,
    }
    rate = discounts.get(code, 0.0)
    return subtotal * (1 - rate)


def main() -> tuple[float, float]:
    """Calculate subtotal and discounted total for predefined items.

    Returns a tuple of (subtotal, discounted_total). No printing is performed.
    """
    items = [
        ("book", 12.50),
        ("pen", 1.20),
        ("notebook", 3.30),
    ]
    subtotal = calculate_subtotal(items)
    discounted = apply_discount(subtotal, "SAVE10")
    return subtotal, discounted

if __name__ == "__main__":
    # No side‑effects when run directly; function can be used programmatically.
    pass
