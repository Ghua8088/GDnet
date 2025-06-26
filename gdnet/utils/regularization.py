from ..gpu import gpu


def l2_regularization(weights, lambda_):
    xp = gpu.xp
    return lambda_ * sum(xp.sum(w**2) for w in weights)

def l1_regularization(weights, lambda_):
    xp = gpu.xp
    return lambda_ * sum(xp.sum(xp.abs(w)) for w in weights)