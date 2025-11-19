import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    poisson_reg = PoissonRegression(step_size=lr)
    #train the model
    poisson_reg.fit(x_train, y_train)

    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = poisson_reg.predict(x_val)

    np.savetxt(pred_path, y_pred)

    plt.figure()
    plt.plot(y_val, y_pred, 'bx')
    margin_1 = (max(y_val) - min(y_val))*0.2
    margin_2 = (max(y_pred) - min(y_pred))*0.2
    x1 = np.arange(min(y_val)-margin_1, max(y_val)+margin_1, 10)
    plt.plot(x1,x1,c='red')
    plt.plot()
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('output/p03d.png')



    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        
        self.theta = np.zeros(n)
        old_theta = np.ones(n)
        while np.linalg.norm(self.theta - old_theta) >= self.eps : 
            old_theta = np.copy(self.theta)
            h_x = np.exp(x@self.theta)
            # batch gradient ascent
            self.theta += self.step_size*((y - h_x)/m).T.dot(x)  
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x@self.theta)
        # *** END CODE HERE ***
