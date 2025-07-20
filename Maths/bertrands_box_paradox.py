import numpy as np

"""
given you draw 1 bill from a box that either
has a 1 dollar bill and a 100 dollar bill
or 
2 100 dollar bills

and you draw a 100 dollar bill first. 
What are the odds you draw a second 100 dollar bill.
"""

def main():
    n = int(1e7)

    boxes = np.array([
        np.array([[1], [100]]),
        np.array([[100], [100]])
    ])
    box_choices = boxes[np.random.randint(0, 2, size=n)]

    first_bills = np.random.randint(0, 2, size=n)

    first_100_mask = box_choices[np.arange(n), first_bills] == 100
    second_100_mask = box_choices[np.arange(n), 1 - first_bills] == 100

    first_draw_was_100_count = np.sum(first_100_mask)
    second_bill_was_100_count = np.sum(first_100_mask & second_100_mask)

    perc = second_bill_was_100_count / first_draw_was_100_count
    print(f"{second_bill_was_100_count:,}/ {first_draw_was_100_count:,} = {perc:.3%}")


if __name__ == "__main__":
    main()
