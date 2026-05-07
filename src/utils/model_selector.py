def select_best_model(scores):
    return min(scores, key=scores.get)