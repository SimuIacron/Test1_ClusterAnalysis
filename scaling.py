from sklearn.preprocessing import StandardScaler

import util


def scaling(data, algorithm="NONE"):
    if algorithm == "STANDARDSCALER":
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler.transform(data)
    elif algorithm == "SCALEMINUSPLUS1":
        return util.rotateNestedLists([util.scale_array_to_minus_plus_1(feature) for feature in
                                       util.rotateNestedLists(data)])
    elif algorithm == "SCALE01":
        return util.rotateNestedLists([util.scale_array_to_01(feature) for feature in
                                       util.rotateNestedLists(data)])
    else:  # algorithm == 'NONE'
        return data
