import pytest
from main import is_palindrome

def test_palindrome_simple():
    assert is_palindrome("madam") is True

def test_palindrome_mixed_case():
    assert is_palindrome("RaceCar") is True

def test_palindrome_with_spaces_and_punct():
    assert is_palindrome("A man, a plan, a canal: Panama") is True

def test_not_palindrome():
    assert is_palindrome("hello") is False

def test_empty_string():
    assert is_palindrome("") is True

def test_single_character():
    assert is_palindrome("x") is True