import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel as C,
                                              RBF, WhiteKernel)


class GPAutoGrad(object):
    """GPAutoGrad implements a GP Regression algorithm which utilizes
    the autogradient to find the derivative of the mean function
    and the derivative of the predictive variance function.

    It inputs a trained model from the scikit-learn library using the
    kernel function: C() * RBF() + WhiteKernel()
    """

    def __init__(self, gp_model):
        self.gp_model = gp_model
        kernel_model = self.gp_model.kernel_
        self.signal_variance = kernel_model.get_params()['k1__k1__constant_value']
        self.length_scale = kernel_model.get_params()['k1__k2__length_scale']
        self.likelihood_variance = kernel_model.get_params()['k2__noise_level']
        self.weights = gp_model.alpha_
        self.x_train = gp_model.X_train_
        self.L = gp_model.L_

    def fit(self):
        return self

    def predict(self, X):

        # kernel matrix
        K = self.rbf_covariance(X, y=self.x_train,
                                length_scale=self.length_scale,
                                scale=self.signal_variance)

        return np.dot(K, self.weights)

    def variance(self, X):

        L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))
        K_inv = np.dot(L_inv, L_inv.T)

        K_trans = self.rbf_covariance(X, y=self.x_train,
                                      length_scale=self.length_scale,
                                      scale=self.signal_variance)

        # compute the variance
        y_var = np.diag(self.rbf_covariance(X, length_scale=self.length_scale,
                                            scale=self.signal_variance)) \
                + self.likelihood_variance
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)

        return y_var

    def mu_grad(self, X, nder=1):

        # Construct the autogradient function for the
        # predictive mean
        mu = lambda x: self.predict(x)

        if nder == 1:
            grad_mu = egrad(mu)
            return grad_mu(X)
        else:
            grad_mu = egrad(egrad(mu))
            return grad_mu(X)

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

    @staticmethod
    def rbf_covariance(X, y=None, scale=1.0, length_scale=1.0):
        if y is None:
            y = X
        D = np.expand_dims(X / length_scale, 1) \
            - np.expand_dims(y / length_scale, 0)
        return scale * np.exp(-0.5 * np.sum(D ** 2, axis=2))


def np_gradient(y_pred, xt, n_points=1000):
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
    # Get sample data
    xtrain, xtest, ytrain, ytest = sample_data()

    # experimental parameters
    seed = 123
    signal_variance = 1.0
    length_scale = 1.0
    noise_likelihood = 0.01

    # Kernel Parameters
    gp_kernel = C(signal_variance) \
                * RBF(length_scale) \
                + WhiteKernel(noise_likelihood)

    # Initialize GP Model
    gp_model = GaussianProcessRegressor(kernel=gp_kernel, random_state=seed)

    gp_model.fit(xtrain, ytrain)

    # Get predictions
    y_pred, y_var = gp_model.predict(xtest, return_std=True)

    # Extract new kernel parameters
    signal_variance = gp_model.kernel_.get_params()['k1__k1__constant_value']
    length_scale = gp_model.kernel_.get_params()['k1__k2__length_scale']
    noise_level = gp_model.kernel_.get_params()['k2__noise_level']

    ##########################
    # First Derivative
    ##########################
    gp_autograd = GPAutoGrad(gp_model)

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

    assert (mu_der2.all() == num_grad2.all())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # ax.scatter(xtrain, ytrain)
    ax.scatter(xtrain, ytrain, color='r', label='Training Points')
    ax.plot(xtest, y_pred, color='k', label='Predictions')
    ax.plot(xtest, mu_der2, color='b', linestyle=":", label='Autograd 2nd Derivative')
    ax.plot(xtest, num_grad2, color='y', linestyle="--", label='Numerical 2nd Derivative')

    ax.legend()

    plt.show()

    return None


if __name__ == '__main__':
    main()
