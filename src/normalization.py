def normalize_min_max(feature, minimum, maximum):
    return (feature - minimum) / (maximum - minimum)

def denormalize_min_max(predictions, minimum, maximum):
    return predictions * (maximum - minimum) + minimum
