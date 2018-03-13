"""
Save data to pickle file
"""
import os
import pickle


def save_object(filename, obj):
    pickle.dump(obj, open(filename+".pkl", 'wb'))


def load_object(filename):
    pickle_fn = filename + ".pkl"

    # load data from pickle file if it exists
    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    return None


def test_save_data():
    print("Test save data")
    test_arr = [1,2,3,4]
    test_fn = os.path.join("data","testfile")
    save_object(test_fn, test_arr)
    loaded_arr = load_object(test_fn)
    assert(loaded_arr == test_arr)
    print("test_arr:",test_arr)
    print("loaded_arr:",test_arr)


def main():
    test_save_data()


if __name__ == "__main__":
    test_save_data()
