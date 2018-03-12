import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_data import load_data


def main():
    print("hello world")
    data = load_data()
    print(data["Seasons"].plot())
    plt.show()
    print(data["Seasons"]["RegionZ"])


if __name__=="__main__":
    main()
