import numpy as np
import read_data

def main():
    print("hello world")
    data = read_data.load_data()
    print(data.keys())

if __name__=="__main__":
    main()
