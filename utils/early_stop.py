class EarlyStopChecker:

    def __init__(self, patience, min_improvement):

        self.patience = patience
        self.min_improvement = min_improvement
        self.best_score = 0
        self.counter = 0

    def __call__(self, score):

        if score - self.best_score >= self.min_improvement:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
