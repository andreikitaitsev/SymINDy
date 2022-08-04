import matplotlib.pyplot as plt

'''Plotting functions and utils.'''
def plot2d(data, fig=None, ax=None, figtitle=None, fontsize=16):
    if fig is None and ax is None:
        fig, axs = plt.subplots(2, figsize=(12,8), sharex=True)
    # plot x
    axs[0].plot(data["time"], data["x_te"][:,0], color="#1874a5", linestyle='dashed')
    axs[0].plot(data["time"], data["x_te"][:,1], color="#1874a5", linestyle='solid')
    axs[0].plot(data["time"], data["x_te_pred"][:,0], color="#e07b39", linestyle='dashed')
    axs[0].plot(data["time"], data["x_te_pred"][:,1], color="#e07b39", linestyle='solid')
    axs[0].set_xlabel('Time', fontsize=fontsize)
    axs[0].set_ylabel('Values', fontsize=fontsize)
    axs[0].set_title('x', fontsize=fontsize)
    axs[0].legend(['x1', 'x2', 'x1_pred', 'x2_pred'])
    axs[0].text(0.1, 0.1, '{}: {:.1f}'.format(data["x_metric"]["name"], data["x_metric"]["value"]))

    # plot xdot
    axs[1].plot(data["time"], data["xdot_te"][:,0], color="#1874a5", linestyle='dashed')
    axs[1].plot(data["time"], data["xdot_te"][:,1], color="#1874a5", linestyle='solid')
    axs[1].plot(data["time"], data["xdot_te_pred"][:,0], color="#e07b39", linestyle='dashed')
    axs[1].plot(data["time"], data["xdot_te_pred"][:,1], color="#e07b39", linestyle='solid')
    axs[1].set_xlabel('Time', fontsize=fontsize)
    axs[1].set_ylabel('Values', fontsize=fontsize)
    axs[1].set_title('xdot', fontsize=fontsize)
    axs[1].legend(['xdot1', 'xdot2', 'xdot1_pred', 'xdot2_pred'])
    axs[1].text(0.1, 0.1, '{}: {:.1f}'.format(data["xdot_metric"]["name"], data["xdot_metric"]["value"]))

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=fontsize+4)

    # x metric
    fig.text(0.5, 0.9,'x: {}: {:.2f}'.format(data["x_metric"]["name"], data["x_metric"]["value"]),
          bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
          ha='center', va='center')
    # x dot metric
    fig.text(0.5, 0.85,'xdot: {}: {:.2f}'.format(data["xdot_metric"]["name"], data["xdot_metric"]["value"]),
          bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
          ha='center', va='center')
          
    return fig, axs


def plot3d(data, fig=None, ax=None, figtitle=None,fontsize=16):
    if fig is None and ax is None:
        fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(
        data["x_te"][:,0],
        data["x_te"][:,1],
        data["x_te"][:,2],
        color="#1874a5")
    ax1.set_title("Original system")
    ax1.set(xlabel="x", ylabel="y", zlabel="z")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(
        data["x_te_pred"][:,0],
        data["x_te_pred"][:,1],
        data["x_te_pred"][:,2],
        color="#e07b39")
    ax2.set_title("Reconstructed system")
    ax2.set(xlabel="x", ylabel="y", zlabel="z")

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=fontsize+4)

    # x metric
    fig.text(0.5, 0.9,'x: {}: {:.2f}'.format(data["x_metric"]["name"], data["x_metric"]["value"]),
          bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
          ha='center', va='center')
    # x dot metric
    fig.text(0.5, 0.85,'xdot: {}: {:.2f}'.format(data["xdot_metric"]["name"], data["xdot_metric"]["value"]),
          bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
          ha='center', va='center')

    return fig, (ax1, ax2)


def plot_compare_sindy_simindy(data, fig=None, ax=None, figtitle=None, fontsize=16):
    if fig is None and ax is None:
        fig, axs = plt.subplots(2, figsize=(12,8), sharex=True)
    # plot original systems and predictions os symindy and pysindy
    axs[0].plot(data["x_te"], color="#1874a5", linestyle='dashed')
    axs[0].plot(data["x_te_pred_symindy"], color="#e07b39", linestyle='dotted')
    axs[0].plot(data["x_te_pred_sindy"], color="#88d269", linestyle='solid')
    axs[0].set_title("x", fointsize=fontsize)
    axs[0].set_xlabel('Time', fontsize=fontsize)
    axs[0].set_ylabel('Values', fontsize=fontsize)
    axs[0].legend(['original','symindy','sindy'])

    axs[1].plot(data["xdot_te"], color="#1874a5", linestyle='dashed')
    axs[1].plot(data["xdot_te_pred_symindy"], color="#e07b39", linestyle='dotted')
    axs[1].plot(data["xdot_te_pred_sindy"], color="#88d269", linestyle='solid')
    axs[1].set_title("x dot", fointsize=fontsize)
    axs[1].set_xlabel('Time', fontsize=fontsize)
    axs[1].set_ylabel('Values', fontsize=fontsize)
    axs[1].legend(['original','symindy','sindy'])
    return fig, axs

def split(data, ratio):
    '''Split the data into the train and test sets
    along the 0th axis.
    Inputs:
        data - np array
        ratio - float
    Outputs:
        data_tr, data_te - splitted data'''
    data_train = data[: int((1-ratio) * len(data))]
    data_test = data[int((1-ratio) * len(data)) :]
    return data_train, data_test
