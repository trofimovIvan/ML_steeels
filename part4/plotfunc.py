import numpy as np
import matplotlib.pyplot as plt
# Function to draw result:
def plot_result(model, X, y_true, metric, title=''):
    """ Args: 
            model -- trained model with fit method.
            X -- feature sample. Must be numpy array or pandas DataFrame
            y -- target sample. True value of target value. Must be numpy array or pandas DataFrame
            title -- title to plot. Must be a string. Default: None.
        Output:
            Draw a plot with dependence of predicted value (y-axis) on the true value (x-axis). Also draw a line y=x.
            """
    y_pred = model.predict(X)
    metric = metric(y_true, y_pred)
    plt.plot(y_true, y_pred, 'o', color='blue')
    x = np.linspace(np.min(y_true), np.max(y_true), 100)
    plt.plot(x, x, color='red')
    plt.title(title + f' MSE = {metric}')
    plt.xlabel('y_true')
    plt.ylabel('y_pred')


def cross_validate_and_plot(model, X, y, cv, metric, title=''):
    """
    Args: 
        model -- model with hyperparameters.
        X -- numpy array. Features
        y -- numpy array. Target value
        title -- title to plot. Must be string. Default -- empty string
    Outputs:
        plot with cross validate. On the title will be shown an average MSE. Also plot y=x line
    """
    MSE_list = []
    predictions = [] # Here will be predictions
    true = [] # Here will be the true value
    best_model = model
    
    # Below is cross validate block. X is divided by a n_splits parts and model fitting only in train sample.
    # Then model is trying to predict test values. This is added to list and shown in the plot.
     
    for train_idx, test_idx in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx] # split data
        best_model.fit(X_train, y_train) # fitting on the train sample
        y_pred = best_model.predict(X_test) # predict test sample
        MSE_list.append(metric(y_test, y_pred)) # mse on the test sample
        predictions.append(y_pred)
        true.append(y_test)
    
    MSE = np.mean(MSE_list)
    n_splits = len(X)
    for i in range(n_splits):
        plt.plot(true[i], predictions[i], 'o', color='blue')
        plt.xlabel('y_true')
        plt.ylabel('y_pred')
        plt.title(title+' CV MSE = {}'.format(MSE))
    x = np.linspace(min(y), max(y), 100)
    plt.plot(x, x, color='red')


def plot_with_err(x, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, "-", **kwargs)
    plt.fill_between(
        x,
        mu - std,
        mu + std,
        edgecolor="none",
        facecolor=lines[0].get_color(),
        alpha=0.2,
    )

def plot_learning_curve(model, X, y, title=''):
    test_sizes = np.linspace(0.1, 1, 5)
    train_sizes = np.linspace(0.1, 1.0, 5)
    N_train, train_scores,  test_scores = learning_curve(model, X, y, train_sizes=train_sizes, cv=5,
                                                           scoring='neg_root_mean_squared_error')
    train_scores *= - 1
    test_scores *= -1
    plt.figure(figsize=(8, 8))
    plot_with_err(N_train, train_scores, label="training scores")
    plot_with_err(N_train, test_scores, label="validation scores")
    plt.xlabel('Train Size')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
