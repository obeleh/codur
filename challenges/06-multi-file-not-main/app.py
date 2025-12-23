from utils import sum_range

def main():
    print(f"sum_range(1, 5) == {sum_range(1, 5)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")