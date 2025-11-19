import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # *** START CODE HERE ***
    # load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # train the model
    gda = GDA()
    gda.fit(x_train, y_train)

    # plot the dataset and the boudary decision
    new_theta = np.zeros(x_train.shape[1]+1)
    new_theta[1 : ] = gda.theta
    new_theta[0] = gda.theta_0     

    # predict the model
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y = gda.predict(x_eval)
    y_pred = y > 0.5

 
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = x.shape
        phi = np.mean(y)
        mu_0 = np.sum((1 - y)[:,None]*x, axis = 0) / np.sum(1-y)
        mu_1 = np.sum(y[:,None]*x, axis = 0) / np.sum(y)
        sigma = np.zeros((n,n))
        for i in range(m):
            if y[i] == 0 :
                sigma += np.outer(x[i,:]-mu_0,x[i,:]-mu_0)
            else :
                sigma += np.outer(x[i,:]-mu_1,x[i,:]-mu_1) 

        sigma_inv = np.linalg.inv(sigma)
        self.theta = sigma_inv.dot(mu_1 - mu_0)
        self.theta_0 = 0.5*(mu_0@sigma_inv@mu_0 - mu_1@sigma_inv@mu_1) -np.log(1/phi - 1)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        h_x = 1/(1 + np.exp(-x@self.theta - self.theta_0))
        return h_x
        # *** END CODE HERE