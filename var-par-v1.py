'''Test the SymINDy with different libraries.
Note, to use scoop launch the script with python -m scoop var-par-v1.py
'''

from main_tmp import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from copy import copy

def plot_myspring(fig, ax, data, title, legend=None):
    ax.plot(data["x_te"][:,0], color='b', linestyle='dashed')
    ax.plot(data["x_te"][:,1], color='b', linestyle='solid')
    ax.plot(data["x_te_pred"][:,0], color='r', linestyle='dashed')
    ax.plot(data["x_te_pred"][:,1], color='r', linestyle='solid')
    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.set_title(title, fontsize=12)
    if legend is None:
        legend=['x test var1', 'x test var2', 'x test predicted var1', 'x test predicted var2']
    fig.legend(legend)
    return fig

def run(var_par, legend=None):
    '''
    Test the reconstruction quality with different parameters.
    Inputs:
        var_par - list of dicts with param name (from SymINDy) as a key and a values for 
            this param as a value (e.g. 
            [{'library_name': 'polynomial'}, {'library_name':'fourier'}, {'library_name': 'generalized'}] )
        title - list of str, title for the axes on the figure
        legends -  list of str, legend for the axes on the figure. Default=None.
    Outputs:
        fig
        x_scores - scores
    '''
    simulator = myspring()
    x, xdot, time = simulator.simulate()
    # split ibservation into train-test sets
    split = lambda x, ratio: (x[:int(ratio*len(x))], x[int(ratio*len(x)):]) if x is not None else (None, None)
    ratio = 0.33
    x_tr, x_te = split(x, ratio)
    xdot_tr, xdot_te = split(xdot, ratio)
    time_tr, time_te = split(time, ratio)
    
    x_te_preds = []
    xdot_te_preds = []
    x_scores=[]
    estimators = []
    
    # create a figure
    fig, axs = plt.subplots(len(var_par), figsize=(16,12), sharex=True, sharey=True)
    fig.subplots_adjust(top = 2)
    fig.tight_layout() 
    
    for n, par_dict in enumerate(var_par):
        # configure SymIDNy
        estimator = SymINDy_class(verbose=True, **par_dict)
        
        # fit on train data
        estimator.fit(x_tr, xdot_tr, time_tr)
        
        # predict test data
        x_te_pred, xdot_te_pred = estimator.predict(x_te[0], time_te)

        # correlation between x_te
        x_score, xdot_score = estimator.score(x_te, x_te_pred, 
            xdot_te, xdot_te_pred)
        x_scores.append(x_score)
        print(f'x_score {x_score:.3f} \n xdot_score {xdot_score}.')

        # collect the data
        data = {"x_te":x_te, "x_te_pred":x_te_pred, "x_score":x_score}

        # plot real and predicted system
        title = 'R2: {:.2f} '.format(x_score) + str(par_dict.keys()) + str(par_dict.values())
        fig = plot_myspring(fig, axs[n], data, title, legend)

        # plot expression tree
        fig_, ax = estimator.plot_trees()

        estimators.append(copy(estimator))
        x_te_preds.append(x_te_pred)
        xdot_te_preds.append(xdot_te_pred)
        del estimator, x_te_pred, xdot_te_pred, x_score, xdot_score, data
    return fig, x_scores

if __name__=='__main__':
    from pathlib import Path
    var_pars = [
        [ {'library_name': 'polynomial'}, {'library_name':'fourier'}, {'library_name': 'generalized'}],
        [ {'ngen': 5}, {'ngen': 20}, {'ngen': 50}],
        [ {'n_individuals': 50}, {'n_individuals': 30}, {'n_individuals': 1000}]
    ]

    fignames = ('var-libs-v1', 'var-ngens-v1', 'var-n_inds-v1')
    for n, var_par in enumerate(var_pars):
        # run symb regr
        fig, x_scores = run(var_par)
        # save fig
        path=Path().cwd().joinpath('figures')
        if not path.is_dir():
            path.mkdir()
        fig.savefig(path.joinpath((fignames[n]+'.png')), dpi=300)
        plt.show()