import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms
import numpy as np

def cov_ellipse(mean, cov, ax, n_std=1.0, edgecolor='blue', facecolor='none', lw=5, **kwargs):
    """
    Plot a covariance ellipse.

    This function plots a covariance ellipse for visualizing the parameter covariance matrix.
    
    Args:
        mean (tuple): Mean of the distribution in the form of (mean_x, mean_y).
        cov (np.ndarray): Covariance matrix of the distribution.
        ax (matplotlib.axes.Axes): Axes object on which to plot the ellipse.
        n_std (float, optional): Number of standard deviations to encompass within the ellipse. Default is 1.0.
        edgecolor (str, optional): Color of the ellipse's edge. Default is 'blue'.
        facecolor (str, optional): Fill color of the ellipse. Default is 'none'.
        lw (float, optional): Linewidth of the ellipse. Default is 5.
        **kwargs: Additional keyword arguments passed to matplotlib.patches.Ellipse.

    Returns:
        matplotlib.patches.Ellipse: The covariance ellipse plotted on the given Axes object.
    """
    
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        lw=lw,
        **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def normal(mean, var, x):
    return np.exp(-(mean-x)**2/var/2)

def CovEllipsePlot(param_names, wave_params, covariance, filename=None):
        fig, axs = plt.subplots(len(param_names),len(param_names), figsize=(20,20))

        #first param index
        for i in range(len(param_names)):
            #second param index
            for j in range(i,len(param_names)):

                if i != j:
                    cov = np.array(((covariance[i][i],covariance[i][j]),(covariance[j][i],covariance[j][j])))
                    #print(cov)
                    mean = np.array((wave_params[param_names[i]],wave_params[param_names[j]]))

                    cov_ellipse(mean,cov,axs[j,i],lw=2,edgecolor='blue')

                    #custom setting the x-y lim for each plot
                    axs[j,i].set_xlim([wave_params[param_names[i]]-2.5*np.sqrt(covariance[i][i]), wave_params[param_names[i]]+2.5*np.sqrt(covariance[i][i])])
                    axs[j,i].set_ylim([wave_params[param_names[j]]-2.5*np.sqrt(covariance[j][j]), wave_params[param_names[j]]+2.5*np.sqrt(covariance[j][j])])

                    axs[j,i].set_xlabel(param_names[i],labelpad=20,fontsize=16)
                    axs[j,i].set_ylabel(param_names[j],labelpad=20,fontsize=16)

                else:
                    mean = wave_params[param_names[i]]
                    var = covariance[i][i]

                    x = np.linspace(mean-3*np.sqrt(var),mean+3*np.sqrt(var))

                    axs[j,i].plot(x,normal(mean,var,x),c='blue')
                    axs[j,i].set_xlim([wave_params[param_names[i]]-2.5*np.sqrt(covariance[i][i]), wave_params[param_names[i]]+2.5*np.sqrt(covariance[i][i])])
                    axs[j,i].set_xlabel(param_names[i],labelpad=20,fontsize=16)
                    if i == j and j == 0:
                        axs[j,i].set_ylabel(param_names[i],labelpad=20,fontsize=16)

        for ax in fig.get_axes():
            ax.label_outer()

        for i in range(len(param_names)):
            for j in range(i+1,len(param_names)):
                fig.delaxes(axs[i,j])

        if filename is None:
            return fig, ax
        else:
            plt.savefig(filename,dpi=300,bbox_inches='tight')
            
def StabilityPlot(deltas,Gammas,param_name=None,filename=None):
    
    plt.figure(figsize=(12,5))
    plt.loglog(deltas,Gammas,'ro-')
    
    if param_name != None:
        plt.xlabel(r'$\Delta\theta_i$',fontsize=12)
    
    plt.ylabel(r'$\left.\langle \frac{\partial h}{\partial \theta_i}\right|\frac{\partial h}{\partial \theta_i}\rangle$',fontsize=14)
    plt.title(r'$\theta_i = $'+f'${param_name}$',fontsize=12)
    plt.grid(True)
    
    if filename != None:
        plt.savefig(filename,dpi=300,bbox_inches='tight')
        
