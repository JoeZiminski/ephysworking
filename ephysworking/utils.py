def save_plot(x, y, xlabel, ylabel, output_filepath, legend=None):  # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(output_filepath)
    plt.close(fig)