import matplotlib.pyplot as plt
import numpy as np

# Data from latest logs
models = ['DIN', 'BST', 'HUG-Unified', 'HUG-Dual (KGA Off)', 'HUG-Dual (KGA On)']
auc = [0.6903, 0.7009, 0.7260, 0.7344, 0.7398]
ap = [0.5392, 0.5443, 0.6001, 0.6114, 0.6225]
loss = [0.6000, 0.5924, 0.5849, 0.5760, 0.5737]
ndcg = [0.7065, 0.8553, 1.0000, 0.9515, 1.0000]

# --- Larger publication-ready styling ---
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

x = np.arange(len(models))
width = 0.35

# Baselines grey, ours colored
model_colors = [
    '#8C8C8C',  # DIN
    '#8C8C8C',  # BST
    '#4C72B0',  # HUG-Unified
    '#55A868',  # HUG-Dual (KGA Off)
    '#2CA02C'   # HUG-Dual (KGA On)
]

def add_labels(ax, rects, fmt='%.3f'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f'{height:{fmt[1:]}}',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=18,
            fontweight='bold'
        )

# --- Figure 1: AUC & AP (RESTORED ORIGINAL COLORS) ---
fig1, ax1 = plt.subplots(figsize=(10, 6))

r1 = ax1.bar(
    x - width/2,
    auc,
    width,
    label='Test AUC',
    color='#4C72B0',
    edgecolor='white',
    capsize=5
)

r2 = ax1.bar(
    x + width/2,
    ap,
    width,
    label='Test AP',
    color='#DD8452',
    edgecolor='white',
    capsize=5
)

ax1.set_ylabel('Metric Score')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15)
ax1.set_ylim(0.50, 0.82)

ax1.legend(loc='upper left', frameon=True)

add_labels(ax1, r1)
add_labels(ax1, r2)

plt.tight_layout()
plt.savefig('figure_auc_ap.pdf')

# --- Figure 2: NDCG ---
fig2, ax2 = plt.subplots(figsize=(9, 6))

r3 = ax2.bar(
    x,
    ndcg,
    color=model_colors,
    edgecolor='white',
    width=0.6
)

ax2.set_ylabel('NDCG@10 Score')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=15)
ax2.set_ylim(0.65, 1.1)

add_labels(ax2, r3)

plt.tight_layout()
plt.savefig('figure_ndcg.pdf')

# --- Figure 3: Log Loss ---
fig3, ax3 = plt.subplots(figsize=(9, 6))

r4 = ax3.bar(
    x,
    loss,
    color=model_colors,
    edgecolor='white',
    width=0.6
)

ax3.set_ylabel('Binary Cross-Entropy')
ax3.set_xticks(x)
ax3.set_xticklabels(models, rotation=15)
ax3.set_ylim(0.55, 0.62)

add_labels(ax3, r4, fmt='%.4f')

plt.tight_layout()
plt.savefig('figure_loss.pdf')

plt.show()