import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    # train the model over the (x_train, t_train)
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    log_reg_t = LogisticRegression()
    log_reg_t.fit(x_train, t_train)

    # test the performance of the model
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    t = log_reg_t.predict(x_test)
    t_pred = t > 0.5
    np.savetxt(pred_path_c, t_pred, fmt="%d")

    # plot the dataset and the decision boundary
    util.plot(x_test, t_test, log_reg_t.theta, 'output/p02c.png')


    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col='y',add_intercept=True )
    
    log_reg_y = LogisticRegression()
    log_reg_y.fit(x_train, y_train)

    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    y = log_reg_y.predict(x_test)
    y_pred = y > 0.5
    np.savetxt(pred_path_d, y_pred, fmt="%d")

    util.plot(x_test, y_test, log_reg_y.theta, 'output/p02d.png')


    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    x_val, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_labeled = x_val[y_val == 1]
    y = log_reg_y.predict(x_labeled)
    y_pred = log_reg_y.predict(x_val)
    

    alpha = np.mean(y)
    
    theta_update = np.copy(log_reg_y.theta)
    theta_update[0] += np.log(2/alpha - 1)
    util.plot(x_test, t_test, theta_update, 'output/p02e.png')

    t_pred_e = y_pred / alpha
    np.savetxt(pred_path_e, t_pred_e > 0.5, fmt='%d')
    # *** END CODER HERE
