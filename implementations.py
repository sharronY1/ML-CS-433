import numpy as np


def least_squares(y, tx):
    """Solve the unregularized least-squares normal equations.

    Args:
        y: One-dimensional target array of shape (N,).
        tx: Design matrix of shape (N, D), including any intercept column.

    Returns:
        A tuple (w, loss), where w has shape (D,) and loss is
        sum((y - tx @ w) ** 2) / (2 * N).
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    error = y - tx @ w
    # half MSE
    loss = np.mean(error**2) / 2
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Solve least squares with an L2 penalty of lambda_ * sum(w ** 2).

    Args:
        y: One-dimensional target array of shape (N,).
        tx: Design matrix of shape (N, D), including any intercept column.
        lambda_: Nonnegative regularization strength applied to all weights.

    Returns:
        A tuple (w, loss), where w has shape (D,) and loss is
        sum((y - tx @ w) ** 2) / (2 * N), excluding the L2 penalty.
    """
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    error = y - tx @ w
    loss = np.mean(error**2) / 2
    return w, loss


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Minimize half mean squared error using full-batch gradient descent.

    Args:
        y: One-dimensional target array of shape (N,).
        tx: Design matrix of shape (N, D), including any intercept column.
        initial_w: Initial weights of shape (D,); copied before updating.
        max_iters: Number of gradient updates.
        gamma: Learning rate.

    Returns:
        A tuple (w, loss), where w has shape (D,) and loss is the final
        sum((y - tx @ w) ** 2) / (2 * N) over all samples.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        gradient = tx.T @ (tx @ w - y) / len(y)
        w = w - gamma * gradient
    error = y - tx @ w
    loss = np.mean(error**2) / 2
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Minimize half mean squared error using single-sample SGD.

    Each update samples one row uniformly with replacement using NumPy's
    global random state. Set np.random.seed before calling for repeatability.

    Args:
        y: One-dimensional target array of shape (N,).
        tx: Design matrix of shape (N, D), including any intercept column.
        initial_w: Initial weights of shape (D,); copied before updating.
        max_iters: Number of single-sample gradient updates.
        gamma: Learning rate.

    Returns:
        A tuple (w, loss), where w has shape (D,) and loss is the final
        sum((y - tx @ w) ** 2) / (2 * N) over all samples.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        # 从 0 到 N-1 中，随机选择一个整数，赋值给 i
        i = np.random.randint(len(y))
        gradient = tx[i] * (tx[i] @ w - y[i])
        w = w - gamma * gradient
    error = y - tx @ w
    loss = np.mean(error**2) / 2
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Minimize mean binary cross-entropy using full-batch gradient descent.

    Args:
        y: One-dimensional binary labels in {0, 1}, of shape (N,).
        tx: Design matrix of shape (N, D), including any intercept column.
        initial_w: Initial weights of shape (D,); copied before updating.
        max_iters: Number of gradient updates.
        gamma: Learning rate.

    Returns:
        A tuple (w, loss), where w has shape (D,) and loss is the final
        mean binary cross-entropy, evaluated with stable logaddexp.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        z = tx @ w
        # 防止overflow的稳定写法，等价的sigmond，可以用对指数性质
        p = np.exp(-np.logaddexp(0, -z))
        gradient = tx.T @ (p - y) / len(y)
        w = w - gamma * gradient
    z = tx @ w
    # y不是1就是0带进去反正东吧
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Minimize mean binary cross-entropy plus lambda_ * sum(w ** 2).

    Full-batch gradient descent penalizes all weights and prints the
    regularized training objective every 200 updates.

    Args:
        y: One-dimensional binary labels in {0, 1}, of shape (N,).
        tx: Design matrix of shape (N, D), including any intercept column.
        lambda_: Nonnegative L2 regularization strength.
        initial_w: Initial weights of shape (D,); copied before updating.
        max_iters: Number of gradient updates.
        gamma: Learning rate.

    Returns:
        A tuple (w, loss), where w has shape (D,) and loss is the final
        mean binary cross-entropy, excluding the L2 penalty.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for step in range(max_iters):
        z = tx @ w
        p = np.exp(-np.logaddexp(0, -z))
        gradient = tx.T @ (p - y) / len(y) + 2 * lambda_ * w
        w = w - gamma * gradient
        if (step + 1) % 200 == 0:
            z = tx @ w
            data_loss = np.mean(np.logaddexp(0, z) - y * z)
            objective = data_loss + lambda_ * np.sum(w**2)
            print("迭代：", step + 1, "训练目标：", objective)
    z = tx @ w
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return w, loss
