import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)


    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    p = len(tau_values)
    mse = np.zeros(p)
    for i in range(p) : 
        lwr = LocallyWeightedLinearRegression(tau_values[i])
        lwr.fit(x_train, y_train)
        y_pred = lwr.predict(x_eval)
        mse[i] = np.mean((y_pred - y_eval)**2)
        print('MSE value for tau = {} : {}'.format(tau_values[i], mse[i]))

        plt.figure()
        plt.plot(x_eval, y_eval, 'bx', linewidth=2)
        plt.plot(x_eval, y_pred, 'ro', linewidth=2)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('output/p05c_tau={}.png'.format(tau_values[i]))

    tau_opt = tau_values[np.argmin(mse)]
    lwr_opt = LocallyWeightedLinearRegression(tau_opt)
    lwr_opt.fit(x_train, y_train)
    y_pred = lwr_opt.predict(x_test)
    mse_test = np.mean((y_pred - y_test)**2)
    print("The MSE on the test dataset : {} for tau_opt = {}".format(mse_test, tau_opt))
    np.savetxt(pred_path, y_pred)



    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***
