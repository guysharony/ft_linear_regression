def normalize_min_max(feature):
    min_val = min(feature)
    max_val = max(feature)
    scaled_feature = (feature - min_val) / (max_val - min_val)
    return scaled_feature, min_val, max_val