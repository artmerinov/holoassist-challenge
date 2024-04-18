class AverageMeter:
    """ 
    Computes and stores a simple average 
    and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class EMAMeter:
    """ 
    Computes and stores an exponential moving average
    and current value.
    """
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.avg = None

    def update(self, value):
        if self.avg is None:
            self.avg = value
        else:
            self.avg = self.alpha * value + (1 - self.alpha) * self.avg