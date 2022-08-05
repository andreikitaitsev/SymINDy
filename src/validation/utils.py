from tkinter import font
import matplotlib.pyplot as plt


color1 = "#1874a5"
color2 = "#e07b39"
color3 = "#88d269"

'''Plotting functions and utils.'''
def plot2d(data, fig=None, ax=None, figtitle=None, fontsize=16):
    if fig is None and ax is None:
        fig, axs = plt.subplots(2, figsize=(12,8), sharex=True)
    
    # plot x
    axs[0].plot(data["time"], data["x_te"][:,0], color=color1, linestyle='dashed')
    axs[0].plot(data["time"], data["x_te"][:,1], color=color1, linestyle='solid')
    axs[0].plot(data["time"], data["x_te_pred"][:,0], color=color2, linestyle='dashed')
    axs[0].plot(data["time"], data["x_te_pred"][:,1], color=color2, linestyle='solid')
    axs[0].set_xlabel('Time', fontsize=fontsize)
    axs[0].set_ylabel('Values', fontsize=fontsize)
    axs[0].set_title('x', fontsize=fontsize)
    axs[0].legend(['x1', 'x2', 'x1_pred', 'x2_pred'])
    axs[0].text(0.05, 0.95, '{}: {:.1f}'.format(data["x_metric"]["name"], data["x_metric"]["value"]),
        verticalalignment='top', horizontalalignment='left',
        transform=axs[0].transAxes)

    # plot xdot
    axs[1].plot(data["time"], data["xdot_te"][:,0], color=color1, linestyle='dashed')
    axs[1].plot(data["time"], data["xdot_te"][:,1], color=color1, linestyle='solid')
    axs[1].plot(data["time"], data["xdot_te_pred"][:,0], color=color2, linestyle='dashed')
    axs[1].plot(data["time"], data["xdot_te_pred"][:,1], color=color2, linestyle='solid')
    axs[1].set_xlabel('Time', fontsize=fontsize)
    axs[1].set_ylabel('Values', fontsize=fontsize)
    axs[1].set_title('xdot', fontsize=fontsize)
    axs[1].legend(['xdot1', 'xdot2', 'xdot1_pred', 'xdot2_pred'])
    axs[1].text(0.05, 0.95, '{}: {:.1f}'.format(data["xdot_metric"]["name"], data["xdot_metric"]["value"]),
        verticalalignment='top', horizontalalignment='left',
        transform=axs[1].transAxes)
    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=fontsize+4)
          
    return fig, axs


def plot3d(data, fig=None, ax=None, figtitle=None,fontsize=16):
    if fig is None and ax is None:
        fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(
        data["x_te"][:,0],
        data["x_te"][:,1],
        data["x_te"][:,2],
        color=color1)
    ax1.set_title("Original system")
    ax1.set_xlabel("x", fontsize=fontsize)
    ax1.set_ylabel("y", fontsize=fontsize)
    ax1.set_zlabel("z",  fontsize=fontsize)


    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(
        data["x_te_pred"][:,0],
        data["x_te_pred"][:,1],
        data["x_te_pred"][:,2],
        color=color2)
    ax2.set_title("Reconstructed system")
    ax2.set_xlabel("x", fontsize=fontsize)
    ax2.set_ylabel("y", fontsize=fontsize)
    ax2.set_zlabel("z",  fontsize=fontsize)
    
    # x metric
    fig.text(0.5, 0.9,'{}(x): {:.2f}'.format(data["x_metric"]["name"], data["x_metric"]["value"]),
          bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
          ha='center', va='center')

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=fontsize+4)

    return fig, (ax1, ax2)


def plot_compare_sindy_simindy(data, fig=None, ax=None, figtitle=None, fontsize=16):
    if fig is None and ax is None:
        fig, axs = plt.subplots(2, figsize=(12,8), sharex=True)
    
    # plot original systems and predictions os symindy and pysindy
    # x
    axs[0].plot(data["x_te"][:,0], color=color1, linestyle='dashed')
    axs[0].plot(data["x_te_pred_symindy"][:,0], color=color2, linestyle='dotted')
    axs[0].plot(data["x_te_pred_sindy"][:,0], color=color3, linestyle='solid')
    axs[0].legend(['original','SymINDy','SINDy'])

    axs[0].plot(data["x_te"][:,1], color=color1, linestyle='dashed')
    axs[0].plot(data["x_te_pred_symindy"][:,1], color=color2, linestyle='dotted')
    axs[0].plot(data["x_te_pred_sindy"][:,1], color=color3, linestyle='solid')

    axs[0].set_title("x")
    
    # x dot
    axs[1].plot(data["xdot_te"][:,0], color=color1, linestyle='dashed')
    axs[1].plot(data["xdot_te_pred_symindy"][:,0], color=color2, linestyle='dotted')
    axs[1].plot(data["xdot_te_pred_sindy"][:, 0], color=color3, linestyle='solid')
    axs[1].legend(['original','SymINDy','SINDy'])

    axs[1].plot(data["xdot_te"][:,1], color=color1, linestyle='dashed')
    axs[1].plot(data["xdot_te_pred_symindy"][:,1], color=color2, linestyle='dotted')
    axs[1].plot(data["xdot_te_pred_sindy"][:, 1], color=color3, linestyle='solid')

    axs[1].set_title("x dot")
    axs[1].set_xlabel('Time', fontsize=fontsize)
    axs[1].set_ylabel('Values', fontsize=fontsize)
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
