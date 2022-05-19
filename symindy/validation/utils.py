import matplotlib.pyplot as plt


'''Plotting functions and utils.'''
def plot_orig_pred(data, fig=None, ax=None, figtitle=None, legend=None, fontsize=16):
    if fig is None and ax is None:
        fig, axs = plt.subplots(2)
    # plot x
    axs[0].plot(data["x_te"][:,0], color='b', linestyle='dashed')
    axs[0].plot(data["x_te"][:,1], color='b', linestyle='solid')
    axs[0].plot(data["x_te_pred"][0], color='r', linestyle='dashed')
    axs[0].plot(data["x_te_pred"][:,1], color='r', linestyle='solid')
    axs[0].set_xlabel('Time', fontsize=fontsize)
    axs[0].set_ylabel('Values', fontsize=fontsize)
    axs[0].set_title('x', fontsize=16)
    axs[0].legend(['x0', 'x1', 'x0_pred', 'x1_pred'])
    axs[0].text(0.1, 0.1, '{}: {:.1f}'.format(data["x_metric"]["name"], data["x_metric"]["value"]))

    # plot xdot
    axs[1].plot(data["xdot_te"][:,0], color='b', linestyle='dashed')
    axs[1].plot(data["xdot_te"][:,1], color='b', linestyle='solid')
    axs[1].plot(data["xdot_te_pred"][0], color='r', linestyle='dashed')
    axs[1].plot(data["xdot_te_pred"][:,1], color='r', linestyle='solid')
    axs[1].set_xlabel('Time', fontsize=fontsize)
    axs[1].set_ylabel('Values', fontsize=fontsize)
    axs[1].set_title('xdot', fontsize=fontsize)
    axs[0].legend(['xdot0', 'xdot1', 'xdot0_pred', 'xdot1_pred'])
    axs[1].text(0.1, 0.1, '{}: {:.1f}'.format(data["xdot_metric"]["name"], data["xdot_metric"]["value"]))

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=fontsize+4)
    fig.tight_layout()
    return fig, axs
