import numpy as np
def plot_learning_curve2(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import mean_absolute_error
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    fig, (ax1,ax2) = plt.subplots(1,2)
    plt.figure()
    #fig.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    ax2.xlabel("Training set size", fontsize=14)
    ax1.xlabel("Training set size", fontsize=14)
    ax1.ylabel("MAE [eV]",fontsize=14)
    ax2.ylabel("MSE [eV]",fontsize=14)
    train_sizes_mae, train_scores_mae, test_scores_mae = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_absolute_error')
    train_sizes_mse, train_scores_mse, test_scores_mse = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_square_error')
    
    train_scores_mean = np.mean(train_scores_mae, axis=1)
    train_scores_std = np.std(train_scores_mae, axis=1)
    test_scores_mean = np.mean(test_scores_mae, axis=1)
    test_scores_std = np.std(test_scores_mae, axis=1)

    train_scores_q25 = np.quantile(train_scores_mae,0.25, axis=1)
    train_scores_q50 = np.quantile(train_scores_mae,0.50, axis=1)
    train_scores_q75 = np.quantile(train_scores_mae, 0.75, axis=1)

    test_scores_q25 = np.quantile(test_scores_mae,0.25, axis=1)
    test_scores_q50 = np.quantile(test_scores_mae,0.50, axis=1)
    test_scores_q75 = np.quantile(test_scores_mae, 0.75, axis=1)
    
###################3
    train_scores_mean = np.mean(train_scores_mse, axis=1)
    train_scores_std = np.std(train_scores_mse, axis=1)
    test_scores_mean = np.mean(test_scores_mse, axis=1)
    test_scores_std = np.std(test_scores_mse, axis=1)
    
    ax1.fill_between(train_sizes, -1*train_scores_q25,
                    -1*train_scores_q75, alpha=0.3,
                     color="r")
    ax1.fill_between(train_sizes, -1*test_scores_q25,
                     -1*test_scores_q75, alpha=0.3, color="b")
    
    
    
    ax1.plot(train_sizes_mae, -train_scores_q50, 'o-', color="r",
             label="Training")
    ax1.plot(train_sizes_mae, -test_scores_q50, 'o-', color="b",
             label="Cross-validation")

    ax1.plot(train_sizes_mae, -train_scores_q50, 'o-', color="r",
             label="Training")
    ax1.plot(train_sizes_mae, -test_scores_q50, 'o-', color="b",
             label="Cross-validation")
    
    train_scores_q25 = np.quantile(train_scores_mse,0.25, axis=1)
    train_scores_q50 = np.quantile(train_scores_mse,0.50, axis=1)
    train_scores_q75 = np.quantile(train_scores_mse, 0.75, axis=1)

    test_scores_q25 = np.quantile(test_scores_mse,0.25, axis=1)
    test_scores_q50 = np.quantile(test_scores_mse,0.50, axis=1)
    test_scores_q75 = np.quantile(test_scores_mse, 0.75, axis=1)
    plt.grid()

    ax2.fill_between(train_sizes_mse, -1*train_scores_q25,
                    -1*train_scores_q75, alpha=0.3,
                     color="r")
    ax2.fill_between(train_sizes_mse, -1*test_scores_q25,
                     -1*test_scores_q75, alpha=0.3, color="b")
    
    
    
    ax2.plot(train_sizes_mse, -train_scores_q50, 'o-', color="r",
             label="Training")
    ax2.plot(train_sizes_mse, -test_scores_q50, 'o-', color="b",
             label="Cross-validation")

    ax2.plot(train_sizes_mse, -train_scores_q50, 'o-', color="r",
             label="Training")
    ax2.plot(train_sizes_mse, -test_scores_q50, 'o-', color="b",
             label="Cross-validation")
    
    plt.legend(loc="best")
    return plt

import numpy as np
def plot_learning_curve_ma_rms(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),savename='name'):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import mean_absolute_error

    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.labelsize'] = 21
    plt.rcParams['lines.linewidth'] =  4
    plt.rcParams['lines.markersize'] = 10
    #plt.rcParams['boxplot.flierprops.markeredgewidth'] = 10


    fig, (ax1) = plt.subplots(1,1,figsize=(11.5,6))
    fig.suptitle(title, fontsize=22,fontweight='bold')
    #plt.figure()
    #fig.title(title)
    if ylim is not None:
        ax1.set_ylim(*ylim)
        #ax2.set_ylim(*ylim)
    #ax2.set_xlabel("Training set size",fontweight='bold')
    ax1.set_xlabel("Training set size",fontweight='bold')
    ax1.set_ylabel("Error / meV $\cdot$ atom$^{-1}}$",fontweight='bold')
    #ax2.set_ylabel("RMSE [eV]",fontweight='bold')
    train_sizes_mae, train_scores_mae, test_scores_mae = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_absolute_error')


    train_sizes_mse, train_scores_mse, test_scores_mse = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_root_mean_squared_error')

    train_scores_mean = np.mean(train_scores_mae, axis=1)
    train_scores_std = np.std(train_scores_mae, axis=1)
    test_scores_mean = np.mean(test_scores_mae, axis=1)
    test_scores_std = np.std(test_scores_mae, axis=1)

    train_scores_q25 = np.quantile(train_scores_mae/201*1000,0.25, axis=1)
    train_scores_q50 = np.quantile(train_scores_mae/201*1000,0.50, axis=1)
    train_scores_q75 = np.quantile(train_scores_mae/201*1000, 0.75, axis=1)

    test_scores_q25 = np.quantile(test_scores_mae/201*1000,0.25, axis=1)
    test_scores_q50 = np.quantile(test_scores_mae/201*1000,0.50, axis=1)
    test_scores_q75 = np.quantile(test_scores_mae/201*1000, 0.75, axis=1)

###################3

    ax1.fill_between(train_sizes_mae, -1*train_scores_q25,
                    -1*train_scores_q75, alpha=0.3, color='tab:orange'
                     )
    ax1.fill_between(train_sizes_mae, -1*test_scores_q25,
                     -1*test_scores_q75, alpha=0.3,color='lightskyblue' )



    ax1.plot(train_sizes_mae, -train_scores_q50, '-',
             label="MAE Training", color='tab:orange')
    ax1.plot(train_sizes_mae, -test_scores_q50, '-',
             label="MAE Testing",color='lightskyblue')
    train_scores_mean = np.mean(train_scores_mse, axis=1)

    train_scores_std = np.std(train_scores_mse, axis=1)
    test_scores_mean = np.mean(test_scores_mse, axis=1)
    test_scores_std = np.std(test_scores_mse, axis=1)

    train_scores_q25 = np.quantile(train_scores_mse/201*1000,0.25, axis=1)
    train_scores_q50 = np.quantile(train_scores_mse/201*1000,0.50, axis=1)

    train_scores_q75 = np.quantile(train_scores_mse/201*1000, 0.75, axis=1)
    test_scores_q25 = np.quantile(test_scores_mse/201*1000,0.25, axis=1)
    test_scores_q50 = np.quantile(test_scores_mse/201*1000,0.50, axis=1)
    test_scores_q75 = np.quantile(test_scores_mse/201*1000, 0.75, axis=1)
    ax1.grid()
    #ax2.grid()
#
    #ax1.fill_between(train_sizes_mse, -1*train_scores_q25,
    #                -1*train_scores_q75, alpha=0.3,
    #                )
    ax1.fill_between(train_sizes_mse, -1*test_scores_q25,
                     -1*test_scores_q75, alpha=0.3,color='tab:blue')



    #ax1.plot(train_sizes_mse, -train_scores_q50, 'o-',label="Training")
    ax1.plot(train_sizes_mse, -test_scores_q50, '-',
             label="RMSE Testing",color='tab:blue')
    ax1.legend(fontsize=16,loc="best")
    fig.tight_layout()
    #fig.savefig(f'/home/riccardo/Dropbox/Descriptors_Paper/ExtreamPres/{savename}', dpi = 300)
    return plt
