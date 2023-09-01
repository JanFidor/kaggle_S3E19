# This is a sample Python script.
import pandas
import pandas as pd


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with open("submission_tft.csv") as f:
        print(sum(1 for line in f))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
