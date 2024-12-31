import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

def plot_sample_efficiency_curve(frames,
                                 point_estimates,
                                 interval_estimates = None,
                                 min_estimates = None,
                                 max_estimates = None,
                                 algorithms=None,
                                 title=None,
                                 out=None, 
                                 color_palette='colorblind',
                                 figsize=(8, 7.5),
                                 xlabel=r'Number of Frames (in millions)',
                                 ylabel='Score',
                                 colors=None,
                                 label_mapping=None,
                                 add_legend=False,
                                 xticks=None,
                                 yticks=None,
                                 xticklabels=None,
                                 yticklabels=None,
                                 labelsize=22,
                                 ticklabelsize=20,
                                 titlesize=26,
                                 linewidth=2,
                                 spinewidth=2,
                                 fill_alpha=0.2,
                                 marker="o",
                                 grid_alpha=0.2,
                                 legendsize='xx-large',
                                 wrect=10,
                                 hrect=10):
    if algorithms is None:
        algorithms = list(point_estimates.keys())
    if colors is None:
        colors = plt.get_cmap(color_palette).colors

    fig, ax = plt.subplots(figsize=figsize)
    # Background color
    fig.patch.set_facecolor("#FFFFFF")
    frames_ = frames.copy()

    # CURVES
    for idx, algorithm in enumerate(algorithms):
        metric_values = point_estimates[algorithm]
        if type(frames) == dict:
            frames_ = frames[algorithm]
        else:
            if metric_values.shape[0] < len(frames):
                frames_ = frames[:metric_values.shape[0]]
            else:
                frames_ = frames.copy()
        if len(metric_values.shape) == 1:
            ax.plot(frames_, metric_values, color=colors[algorithm], marker=marker, linewidth=linewidth, 
                    label=label_mapping[algorithm] if label_mapping else algorithm)
        elif len(metric_values.shape) == 2:
            for i in range(metric_values.shape[1]):
                ax.plot(frames_, metric_values[:, i], color=colors[algorithm], marker=marker, linewidth=linewidth, 
                        label=label_mapping[algorithm] if label_mapping else algorithm)
        
        # CI or std if provided
        if interval_estimates is not None:
            interval_values = interval_estimates[algorithm]
            if len(interval_estimates[algorithm].shape) == 2:
                lower, upper = interval_values
            else:
                lower = metric_values - interval_values
                upper = metric_values + interval_values
            ax.fill_between(frames_, y1=lower, y2=upper, color=colors[algorithm], alpha=fill_alpha)

        # Min and max interval if provided
        if min_estimates is not None and max_estimates is not None:
            ax.fill_between(frames_, y1=min_estimates[algorithm], y2=max_estimates[algorithm], color=colors[algorithm], alpha=fill_alpha)

    # FORMATTING
    # X label, Y label, and title
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    if title:
        ax.set_title(title, fontsize=titlesize)
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(spinewidth)
    ax.spines['bottom'].set_linewidth(spinewidth)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines['left'].set_position(('outward', hrect))
    ax.spines['bottom'].set_position(('outward', wrect))

    # Customizable ticks and tick labels
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)

    # Grid
    ax.grid(True, alpha=grid_alpha)

    # Legend
    if add_legend:
        leg = ax.legend(fontsize=legendsize)
        leg.get_title().set_fontsize(12)
    
    # Save
    if out is not None:
        plt.savefig(out, bbox_inches='tight')

    # Show
    plt.show()
    return

# run-ids (algos), labels, colors
ALGOS = ["ppo", "pos_enc", "16stack", "4stack", "gru", "gru_rec", "trxl", "trxl_rec", "gtrxl_b0", "gtrxl_b0_rec", "gtrxl_b2", "gtrxl_b2_rec", "trxl_lr", "trxl_rec_lr", "trxl_qpos", "trxl_gt", "trxl_gt_qpos_lr", "lstm", "trxl_lr_learned", "trxl_lr_relative", "gru_384"]
LABELS = ["PPO without Memory", "Relative Positional Encoding", "16 Frames Stack", "4 Frames Stack", "GRU", "GRU + Obs. Rec.", "TrXL", "TrXL + Obs. Rec.", "GTrXL b0", "GTrXL b0 + Obs. Rec.", "GTrXL b2", "GTrXL b2 + Obs. Rec.", "TrXL + LR", "TrXL + LR + Obs. Rec.", "TrXL + QPos", "TrXL + GT", "TrXL + GT + QPos + LR", "LSTM", "TrXL + LR + LPE", "TrXL + LR + RPE", "GRU 384"]
LABEL_MAPPING = dict(zip(ALGOS, LABELS))
LABEL_MAPPING["gru_25"] = LABEL_MAPPING["gru"]
LABEL_MAPPING["trxl_25"] = LABEL_MAPPING["trxl"]
LABEL_MAPPING["gru_rec_25"] = LABEL_MAPPING["gru_rec"]
LABEL_MAPPING["trxl_rec_25"] = LABEL_MAPPING["trxl_rec"]
# colors_ = sns.color_palette("bright")
# colors = sns.color_palette("tab20", n_colors=len(algos))
# colors[-1] = colors_[8]
b_c = sns.color_palette('bright', 16)
d_c = sns.color_palette('dark', 16)
c_c = sns.color_palette('colorblind', 16)
COLORS = [b_c[0], b_c[4], b_c[1], d_c[7], b_c[2], d_c[2], b_c[3], d_c[3], b_c[0], d_c[0], b_c[4], d_c[4], b_c[4], d_c[4], b_c[1], d_c[7], b_c[8], b_c[8], b_c[9], b_c[0], b_c[2]]
COLOR_MAPPING = dict(zip(ALGOS, COLORS[:len(ALGOS)]))
COLOR_MAPPING["gru_25"] = COLOR_MAPPING["gru"]
COLOR_MAPPING["trxl_25"] = COLOR_MAPPING["trxl"]
COLOR_MAPPING["gru_rec_25"] = COLOR_MAPPING["gru_rec"]
COLOR_MAPPING["trxl_rec_25"] = COLOR_MAPPING["trxl_rec"]

def load_and_process_experiment_data(file_path, data_key):
    files = [file_path + f for f in os.listdir(file_path) if f.endswith(".res")]
    print(files)
    loaded_data = [pickle.load(open(f, "rb")) for f in files]

    # Get shape of data
    num_runs = len(loaded_data)
    num_checkpoints = loaded_data[0].shape[0]
    num_repetitions = loaded_data[0].shape[1]
    num_episodes = loaded_data[0].shape[2]
    value_keys = [*loaded_data[0][0,0,0]]
    
    if data_key not in value_keys:
        raise ValueError(f"Data key {data_key} not found in loaded data. Available keys: {value_keys}")

    # create a target array with the correct shape
    raw_data_array = np.zeros((num_runs, num_checkpoints, num_repetitions, num_episodes))

    # fill the target array with the data
    for n in range(num_runs):
        for i in range(num_checkpoints):
            for j in range(num_repetitions):
                for k in range(num_episodes):
                    raw_data_array[n,i,j,k] = loaded_data[n][i,j,k][data_key]

    # swap num_runs and num_checkpoints dimension
    raw_data_array = np.swapaxes(raw_data_array, 0, 1)

    # flatten episode and repetition dimension
    raw_data_array = raw_data_array.reshape((num_checkpoints, num_runs, num_repetitions * num_episodes))

    return raw_data_array

def arange_frames(num_checkpoints, skip=1):
    num_workers = 32
    num_steps = 512
    checkpoint_interval = 500
    frames = np.asarray(list(range(0, num_checkpoints)))
    frames = frames * (num_workers*num_steps*checkpoint_interval) / 1e6
    frames = frames[::skip]
    return frames