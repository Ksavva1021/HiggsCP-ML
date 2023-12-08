import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep  # Assuming you have the mplhep package installed

def PlotHistograms(dataset_1, dataset_2, label_1, label_2, xlabel, ylabel, xmin, xmax, bins, savefile):
    # Create a figure with two subplots side by side
    f, axs = plt.subplots(1, 2, figsize=(20, 12), sharey=False)

    # Labeling for CMS style (mplhep package)
    for ax in axs:
        hep.cms.label(rlabel="", ax=ax)

    # Histogram of dataset_1
    h1, edges = np.histogram(dataset_1, range=(xmin, xmax), bins=bins)
    # Histogram of dataset_2 with the same bin edges as dataset_1
    h2, edges = np.histogram(dataset_2, edges)

    # Calculate bin centers
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Plotting histogram for dataset_1 with error bars
    axs[0].errorbar(bin_centers, h1, np.sqrt(h1), fmt="none", ecolor="red")
    # Plotting histogram for dataset_1
    axs[0].hist(edges[:-1], edges, histtype="step", color="red", label=label_1, weights=h1)
    # Plotting histogram for dataset_2
    axs[0].hist(edges[:-1], edges, histtype="step", color="black", label=label_2, weights=h2)
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlabel(xlabel)
    axs[0].legend(loc='best')

    # Calculate the difference between dataset_1 and dataset_2
    difference = dataset_1 - dataset_2
    # Histogram of the difference
    h_diff, edges_diff = np.histogram(difference, range=(xmin, xmax), bins=bins)

    # Plotting histogram for the difference
    axs[1].hist(edges_diff[:-1], edges_diff, histtype="step", color="green", label='Difference', weights=h_diff)
    axs[1].set_ylabel("Difference", labelpad=10)  # Increase labelpad to avoid overlap
    axs[1].set_xlabel(xlabel)

    # Adjust layout to prevent overlapping
    f.tight_layout()
    # Save the figure as a PDF file
    f.savefig(savefile + ".pdf")
    plt.close()

def AcoPlot(bins, r, x, weights, title, savefig):
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Histogram for Scalar weights
    hist1, bin_edges, _ = ax.hist(x, bins=bins, range=r, color='b', weights=weights["wt_cp_sm"].values,
                                  label='Scalar', histtype='step', lw=1.5)
    # Histogram for Pseudoscalar weights
    hist2, bin_edges, _ = ax.hist(x, bins=bins, range=r, color='r', weights=weights["wt_cp_ps"].values,
                                  label='Pseudoscalar', histtype='step', lw=1.5)
    # Histogram without weights
    hist3, bin_edges, _ = ax.hist(x, bins=bins, range=r, color='g', label='No Weights', histtype='step', lw=1, linestyle="dashed")

    # Calculate asymmetry between Scalar and Pseudoscalar
    asymmetry_overall = (1 / bins) * (np.sum(abs(hist1 - hist2) / (hist1 + hist2)))
    # Add text annotation to the plot
    ax.text(0.5, 0.3, "Asymmetry: {:.3f}".format(asymmetry_overall), ha='center', va='center', transform=ax.transAxes)

    # Set plot title, xlabel, ylabel
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('Acolanarity angle $\phi_A$')
    ax.set_ylabel('Events')

    # Add legend to the lower center of the plot
    ax.legend(loc='lower center')

    fig.tight_layout()
    # Save the figure
    fig.savefig(savefig)

