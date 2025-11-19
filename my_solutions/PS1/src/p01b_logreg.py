import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # trains the model 
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    # plot the dataset and the boudary decision
    util.plot(x_train, y_train, log_reg.theta, 'output/p01b_{}.png'.format(pred_path[-5]))

    # predicts for the eval dataset
    y = log_reg.predict(x_eval)
    y_pred = y > 0.5 

    np.savetxt(pred_path, y_pred, fmt = "%d")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        #computes the logistic function : h_x
        m, n = x.shape

        
        self.theta = np.zeros(n)
        old_theta = np.ones(n)

        while np.linalg.norm(self.theta - old_theta) >= self.eps : 
            #initialize the parameters
            h_x = np.zeros(m)
            grad_J = np.zeros(n)
            H = np.zeros((n,n))
            
            #computes the logistic function : h_x
            h_x = 1/(1 + np.exp(-x@self.theta)) 

            #computes the gradient of J : grand_J
            diff = h_x - y
            grad_J = x.T@diff / m

            #computes the hessian's inverse : H_inv
            g = h_x*(1 - h_x)
            for i in range(m):
                H += g[i]*(np.outer(x[i,:].T, x[i,:]))
            H = H/m
            H_inv = np.linalg.inv(H)

            #update theta
            old_theta = np.copy(self.theta)
            self.theta -= H_inv@grad_J
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]

        h_x = np.zeros(m)
        h_x = 1/(1 + np.exp(-x@self.theta)) 
        return h_x
        
        # *** END CODE HERE ***
