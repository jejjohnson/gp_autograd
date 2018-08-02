import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad, value_and_grad
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from operator import itemgetter
from data import example_1d


# TODO: Draw Samples from Prior
# TODO: Draw Samples from Posterior
# Link: https://github.com/paraklas/GPTutorial/blob/master/code/gaussian_process.py
# TODO: Generalize for any kernel methods
# TODO: Add Comments

# TODO: Constrained Optimization Function
# TODO: Multiple Iterations for finding best weights for the derivative
# Link: https://github.com/scikit-learn/scikit-learn/blob/f0ab589f/sklearn/gaussian_process/gpr.py#L451


class GaussianProcess(BaseEstimator, RegressorMixin):
    def __init__(self, jitter=1e-8, random_state=None):
        self.jitter = jitter
        self.random_state = random_state

    def init_theta(self):
        """Initializes the hyperparameters."""
        signal_variance = 1.0
        length_scale = np.ones(self.X_train_.shape[1])
        noise_likelihood = 0.01
        theta = np.array([signal_variance, noise_likelihood, length_scale])
        return np.log(theta)

    def fit(self, X, y):

        self.X_train_ = X
        self.y_train_ = y

        # initial hyper-parameters
        theta0 = self.init_theta()

        # minimize the objective function
        best_params = minimize(value_and_grad(self.log_marginal_likelihood), theta0, jac=True,
                               method='L-BFGS-B')
        # Gather hyper parameters
        signal_variance, noise_likelihood, length_scale = \
            self._get_kernel_params(best_params.x)

        self.signal_variance = np.exp(signal_variance)
        self.noise_likelihood = np.exp(noise_likelihood)
        self.length_scale = np.exp(length_scale)

        # Calculate the weights
        K = self.rbf_covariance(X, length_scale=self.length_scale,
                                signal_variance=self.signal_variance)
        K += self.noise_likelihood * np.eye(K.shape[0])
        L = np.linalg.cholesky(K + self.jitter * np.eye(K.shape[0]))
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y))

        self.weights = weights
        self.L = L
        self.K = K

        L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))

        self.K_inv = np.dot(L_inv, L_inv.T)

        return self

    def log_marginal_likelihood(self, theta):
        x_train = self.X_train_
        y_train = self.y_train_

        if np.ndim == 1:
            y_train = y_train[:, np.newaxis]

        # Gather hyper parameters
        signal_variance, noise_likelihood, length_scale = \
            self._get_kernel_params(theta)
        signal_variance = np.exp(signal_variance)
        noise_likelihood = np.exp(noise_likelihood)
        length_scale = np.exp(length_scale)

        n_samples = x_train.shape[0]

        # train kernel
        K = self.rbf_covariance(x_train, length_scale=length_scale,
                                signal_variance=signal_variance)
        K += noise_likelihood * np.eye(n_samples)
        L = np.linalg.cholesky(K + self.jitter * np.eye(n_samples))
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, weights)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= (K.shape[0] / 2) * np.log(2 * np.pi)

        log_likelihood = log_likelihood_dims.sum(-1)

        return -log_likelihood

    def predict(self, X, return_std=False):

        # Train test kernel
        K_trans = self.rbf_covariance(X, self.X_train_,
                                      length_scale=self.length_scale,
                                      signal_variance=self.signal_variance)

        pred_mean = np.dot(K_trans, self.weights)

        if not return_std:
            return pred_mean
        else:
            return pred_mean, self.variance(X, K_trans=K_trans)

    def variance(self, X, K_trans=None):

        if K_trans is None:
            K_trans = self.rbf_covariance(X, y=self.X_train_,
                                          length_scale=self.length_scale,
                                          signal_variance=self.signal_variance)

        # compute the variance
        y_var = np.diag(self.rbf_covariance(X, length_scale=self.length_scale,
                                            signal_variance=self.signal_variance)) \
                + self.noise_likelihood
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self.K_inv), K_trans)

        return y_var

    def _get_kernel_params(self, theta):

        signal_variance = theta[0]
        noise_likelihood = theta[1] + self.jitter
        length_scale = theta[2:]

        return signal_variance, noise_likelihood, length_scale

    def rbf_covariance(self, X, y=None, signal_variance=1.0, length_scale=1.0):

        if y is None:
            y = X

        D = np.expand_dims(X / length_scale, 1) - np.expand_dims(y / length_scale, 0)

        return signal_variance * np.exp(-0.5 * np.sum(D ** 2, axis=2))

    def mu_grad(self, X, nder=1, return_std=False):

        # Construct the autogradient function for the
        # predictive mean
        mu = lambda x: self.predict(x)
        grad_mu = egrad(mu)

        mu_sol = X

        while nder:
            mu_sol = grad_mu(mu_sol)
            nder -= 1

        if return_std:
            return mu_sol, self.sigma_grad(X, nder=nder)
        else:
            return mu_sol




        # if not return_std:
        #     return grad_mu(X)
        # else:
        #     return grad_mu(X), self.sigma_grad(X, nder=1)
        # else:
        #     grad_mu = egrad(egrad(mu))
        #     if not return_std:
        #         return grad_mu(X)
        #     else:
        #         return grad_mu(X), self.sigma_grad(X, nder=2)

    def sigma_grad(self, X, nder=1):

        # Construct the autogradient function for the
        # predictive variance
        sigma = lambda x: self.variance(x)

        if nder == 1:
            grad_var = egrad(sigma)
            return grad_var(X)
        else:
            grad_var = egrad(egrad(sigma))
            return grad_var(X)


class GaussianProcessError(BaseEstimator, RegressorMixin):
    def __init__(self, jitter=1e-8, x_covariance=None, random_state=None, n_iters=3):
        self.jitter = jitter
        self.x_covariance = x_covariance
        self.random_state = random_state
        self.n_ters= n_iters

    def init_theta(self):
        """Initializes the hyperparameters."""
        signal_variance = np.log(1.0)
        length_scale = np.log(np.ones(self.X_train_.shape[1]))
        noise_likelihood = np.log(0.01)

        theta = np.hstack([signal_variance, noise_likelihood, length_scale])

        return theta

    def fit(self, X, y):

        self.X_train_ = X
        self.y_train_ = y

        if self.x_covariance is None:
            self.x_covariance = 0.0 * self.X_train_.shape[1]
        if np.ndim(self.x_covariance) == 1:
            self.x_covariance = np.array([self.x_covariance])

        # initial hyper-parameters
        theta0 = self.init_theta()

        # Calculate the initial weights


        self.derivative = np.ones(self.X_train_.shape)
        print(self.derivative[:10, :10])
        # minimize the objective function
        optima = [minimize(value_and_grad(self.log_marginal_likelihood), theta0, jac=True,
                               method='L-BFGS-B')]
        fig, ax = plt.subplots()

        ax.scatter(self.X_train_, self.derivative)

        if self.n_ters is not None:

            for iteration in range(self.n_ters):
                print(theta0)
                # Find the minimum
                iparams = minimize(value_and_grad(self.log_marginal_likelihood), theta0, jac=True,
                             method='L-BFGS-B')
                print(iparams)
                # extract best values
                signal_variance, noise_likelihood, length_scale = \
                    self._get_kernel_params(iparams.x)

                # Recalculate the derivative
                K = self.rbf_covariance(self.X_train_, length_scale=np.exp(length_scale),
                                        signal_variance=np.exp(signal_variance))
                K += np.exp(noise_likelihood) * np.eye(K.shape[0])
                L = np.linalg.cholesky(K + self.jitter * np.eye(K.shape[0]))
                iweights = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train_))

                self.derivative = self.weights_grad(self.X_train_, iweights,
                                                    np.exp(length_scale), np.exp(signal_variance))
                print(self.derivative[:10, :10])
                ax.scatter(self.X_train_, self.derivative)
                # make a new theta
                theta0 = np.hstack([signal_variance, noise_likelihood, length_scale])
        plt.show()
        print()
        print(optima)
        lml_values = list(map(itemgetter(1), optima))
        best_params = optima[np.argmin(lml_values)][0]

        print(best_params)
        # Gather hyper parameters
        signal_variance, noise_likelihood, length_scale = \
            self._get_kernel_params(best_params)

        self.signal_variance = np.exp(signal_variance)
        self.noise_likelihood = np.exp(noise_likelihood)
        self.length_scale = np.exp(length_scale)

        # Calculate the weights
        K = self.rbf_covariance(X, length_scale=self.length_scale,
                                signal_variance=self.signal_variance)
        K += self.noise_likelihood * np.eye(K.shape[0])
        L = np.linalg.cholesky(K + self.jitter * np.eye(K.shape[0]))
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y))

        self.weights = weights
        self.L = L
        self.K = K

        L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))

        self.K_inv = np.dot(L_inv, L_inv.T)

        return self

    def log_marginal_likelihood(self, theta):
        x_train = self.X_train_
        y_train = self.y_train_

        if np.ndim == 1:
            y_train = y_train[:, np.newaxis]

        # Gather hyper parameters
        signal_variance, noise_likelihood, length_scale = \
            self._get_kernel_params(theta)
        signal_variance = np.exp(signal_variance)
        noise_likelihood = np.exp(noise_likelihood)
        length_scale = np.exp(length_scale)

        # Calculate the derivative
        # derivative_term = np.diag(np.einsum("ij,ij->i", np.dot(self.derivative, self.x_covariance), self.derivative))
        derivative_term = np.dot(self.derivative, np.dot(self.x_covariance, self.derivative.T))
        n_samples = x_train.shape[0]

        # Calculate derivative of the function

        # train kernel
        K = self.rbf_covariance(x_train, length_scale=length_scale,
                                signal_variance=signal_variance)
        K += noise_likelihood * np.eye(n_samples)
        K += derivative_term
        L = np.linalg.cholesky(K + self.jitter * np.eye(n_samples))
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, weights)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= (K.shape[0] / 2) * np.log(2 * np.pi)

        log_likelihood = log_likelihood_dims.sum(-1)

        return -log_likelihood

    def predict(self, X, return_std=False):

        # Train test kernel
        K_trans = self.rbf_covariance(X, self.X_train_,
                                      length_scale=self.length_scale,
                                      signal_variance=self.signal_variance)

        pred_mean = np.dot(K_trans, self.weights)

        if not return_std:
            return pred_mean
        else:
            return pred_mean, self.variance(X, K_trans=K_trans)

    def variance(self, X, K_trans=None):

        if K_trans is None:
            K_trans = self.rbf_covariance(X, y=self.X_train_,
                                          length_scale=self.length_scale,
                                          signal_variance=self.signal_variance)

        # compute the variance
        y_var = np.diag(self.rbf_covariance(X, length_scale=self.length_scale,
                                            signal_variance=self.signal_variance)) \
                + self.noise_likelihood
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self.K_inv), K_trans)

        return y_var

    def _get_kernel_params(self, theta):

        signal_variance = theta[0]
        noise_likelihood = theta[1] + self.jitter
        length_scale = theta[2:self.X_train_.shape[1] + 2]
        # print(length_scale.shape, init_weights.shape)

        return signal_variance, noise_likelihood, length_scale

    def rbf_covariance(self, X, y=None, signal_variance=1.0, length_scale=1.0):

        if y is None:
            y = X

        D = np.expand_dims(X / length_scale, 1) - np.expand_dims(y / length_scale, 0)

        return signal_variance * np.exp(-0.5 * np.sum(D ** 2, axis=2))

    def predict_weights(self, X, weights, length_scale, signal_variance):
        # Train test kernel
        K_trans = self.rbf_covariance(X,
                                      length_scale=length_scale,
                                      signal_variance=signal_variance)

        pred = np.dot(K_trans, weights)

        return pred

    def weights_grad(self, X, weights, length_scale, signal_variance):

        mu = lambda x: self.predict_weights(x, weights, length_scale, signal_variance)

        grad_mu = egrad(mu)
        return grad_mu(X)

    def mu_grad(self, X, nder=1, return_std=False):

        # Construct the autogradient function for the
        # predictive mean
        mu = lambda x: self.predict(x)

        if nder == 1:
            grad_mu = egrad(mu)

            if not return_std:
                return grad_mu(X)
            else:
                return grad_mu(X), self.sigma_grad(X, nder=1)
        else:
            grad_mu = egrad(egrad(mu))
            if not return_std:
                return grad_mu(X)
            else:
                return grad_mu(X), self.sigma_grad(X, nder=2)

    def sigma_grad(self, X, nder=1):

        # Construct the autogradient function for the
        # predictive variance
        sigma = lambda x: self.variance(x)

        if nder == 1:
            grad_var = egrad(sigma)
            return grad_var(X)
        else:
            grad_var = egrad(egrad(sigma))
            return grad_var(X)


def np_gradient(y_pred, xt):
    return np.gradient(y_pred.squeeze(), xt.squeeze(), edge_order=2)[:, np.newaxis]


def sample_data():
    """Gets some sample data."""
    d_dimensions = 1
    n_samples = 20
    noise_std = 0.1
    seed = 123
    rng = np.random.RandomState(seed)

    n_train = 20
    n_test = 1000
    n_train = 20
    n_test = 1000
    xtrain = np.linspace(-4, 5, n_train).reshape(n_train, 1)
    xtest = np.linspace(-4, 5, n_test).reshape(n_test, 1)

    f = lambda x: np.sin(x) * np.exp(0.2 * x)
    ytrain = f(xtrain) + noise_std * rng.randn(n_train, 1)
    ytest = f(xtest)

    return xtrain, xtest, ytrain, ytest


def main():
    test_error()
    pass


def test_error():
    X, y, error_params = example_1d()
    # Initialize GP Model
    gp_autograd = GaussianProcessError()


    # Fit GP Model
    gp_autograd.fit(X['train'], y['train'])
    # Make Predictions
    y_pred, y_var = gp_autograd.predict(X['test'], return_std=True)
    pass


def test_original():
    # Get sample data
    xtrain, xtest, ytrain, ytest = sample_data()

    # Initialize GP Model
    gp_autograd = GaussianProcess()

    # Fit GP Model
    gp_autograd.fit(xtrain, ytrain)

    # Make Predictions
    y_pred, y_var = gp_autograd.predict(xtest, return_std=True)


    ##########################
    # First Derivative
    ##########################


    # Autogradient
    mu_der = gp_autograd.mu_grad(xtest)

    # # Numerical Gradient
    num_grad = np_gradient(y_pred, xtest)

    assert (mu_der.shape == num_grad.shape)

    # Plot Data
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(xtrain, ytrain, color='r', label='Training Noise')
    ax.plot(xtest, y_pred, color='k', label='Predictions')
    ax.plot(xtest, mu_der, color='b', linestyle=":", label='Autograd 1st Derivative')
    ax.plot(xtest, num_grad, color='y', linestyle="--", label='Numerical Derivative')

    ax.legend()
    plt.show()

    ############################
    # 2nd Derivative
    ############################
    mu_der2 = gp_autograd.mu_grad(xtest, nder=2)
    num_grad2 = np_gradient(num_grad, xtest)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # ax.scatter(xtrain, ytrain)
    ax.scatter(xtrain, ytrain, color='r', label='Training Points')
    ax.plot(xtest, y_pred, color='k', label='Predictions')
    ax.plot(xtest, mu_der2, color='b', linestyle=":", label='Autograd 2nd Derivative')
    ax.plot(xtest, num_grad2, color='y', linestyle="--", label='Numerical 2nd Derivative')

    ax.legend()

    plt.show()

    assert (mu_der2.all() == num_grad2.all())

    return None

if __name__ == '__main__':
    main()
