def scaleArrayTo01(array):
    max_v = max(array)
    min_v = min(array)
    if max_v != 0:
        scaled_array = [((value - min_v) / max_v) for value in array]
    else:
        scaled_array = [0] * len(array)

    return scaled_array

def rotateNestedLists(nested_list):
    return list(map(list, zip(*nested_list)))
