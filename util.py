# scales given array values to a scale of 0 to 1
# example: [3,5,7] --> [0, 0.5, 1]
def scaleArrayTo01(array):
    max_v = max(array)
    min_v = min(array)
    if max_v != 0:
        scaled_array = [((value - min_v) / (max_v-min_v)) for value in array]
    else:
        scaled_array = [0] * len(array)

    return scaled_array


# rotates given nested list
# example: [[1,2], [3,4], [5,6]] --> [[1,3,5], [2,4,6]]
def rotateNestedLists(nested_list):
    return list(map(list, zip(*nested_list)))
