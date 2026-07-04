from context import *
from matplotlib.ticker import MaxNLocator, FixedLocator, LogLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

color_ax1 = "#d8b365"
color_ax2 = "#5ab4ac"


mpl.rcParams.update({
    "figure.facecolor": "white",
    "figure.dpi": 120,

    "axes.facecolor": "#f2f2f2",    
    "axes.edgecolor": "black",
    "axes.linewidth": 1.5,           
    "axes.grid": True,
    "axes.axisbelow": True,

    "grid.color": "white",
    "grid.linewidth": 1.2,
    "grid.linestyle": "-",    

    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,

    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,

    "lines.linewidth": 2.5,
    "lines.markersize": 7,

    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})


rng = np.random.default_rng(42)
n_samples = 5000000
prefix = "1458674"
inner_iters_list = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]


V_gt, _, _, F_gt, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-inner-iters" / f"{prefix}_gt.obj"))

V_ours = {}
F_ours = {}
hdff_dists_ours = []
chamfer_dists_ours = []

for inner_iters in inner_iters_list:
    V, _, _, F, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-inner-iters" / f"{prefix}_in{inner_iters}_ours.obj"))
    V_ours[inner_iters] = V
    F_ours[inner_iters] = F

    # Triangulate faces if needed
    F = utility.triangulate_quads(V, F)

    chamfer_dist = utility.compute_chamfer_dist(V, F, V_gt, F_gt, rng=rng, n_samples=n_samples)
    hdff_dist = utility.compute_hausdorff_distance(V, F, V_gt, F_gt, rng=rng)
    chamfer_dists_ours.append(chamfer_dist)
    hdff_dists_ours.append(hdff_dist)

# Read timings
df = pd.read_csv(os.path.join(ROOT, 'results/ablation-inner-iters/timings_inner_iters.csv'))
# Ensure correct ordering and filtering
df = df[df["inner_iters"].isin(inner_iters_list)]
df = df.sort_values("inner_iters")

inner_iters = np.array(inner_iters_list)
chamfer = np.array(chamfer_dists_ours)
runtime = df["time_seconds"].astype(float).to_numpy()

fig, ax1 = plt.subplots(figsize=(8, 6))

markers = [
    "o",   # circle
    "s",   # square
    "D",   # diamond
    "^",   # triangle up (use only one triangle)
    "P",   # filled plus
    "X",   # filled x
    "*",   # star
    "h",   # hexagon
    "8",   # octagon
    "p",   # pentagon
]
assert len(markers) >= len(inner_iters)

# Left y-axis: Chamfer distance
# Chamfer line
ax1.plot(
    inner_iters,
    chamfer,
    linestyle="-",
    color=color_ax1,
    marker=None,
)

# Plot custom markers
for x, y, m in zip(inner_iters, chamfer, markers):
    ax1.plot(
        x,
        y,
        marker=m,
        linestyle="-",
        color=color_ax1,
        markersize=7,
    )
ax1.set_xlabel("Inner iterations")
ax1.set_xscale("log")
#ax1.set_yscale("log")
ax1.set_ylabel("Chamfer distance")
ax1.tick_params(axis="y", color=color_ax1)

# Right y-axis: Runtime
ax2 = ax1.twinx()
# Runtime line
ax2.plot(
    inner_iters,
    runtime,
    linestyle="--",
    color=color_ax2,
    marker=None,
)
# Plot custom markers
for x, y, m in zip(inner_iters, runtime, markers):
    ax2.plot(
        x,
        y,
        marker=m,
        linestyle="--",
        color=color_ax2,
        markersize=7,
    )
ax2.set_ylabel("Runtime (s)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.tick_params(axis="y", color=color_ax2)

# Align y ticks on both axes
n_ticks = 7

# Use the ticks on ax2 as reference
ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=n_ticks))
y2_ticks = ax2.get_yticks()

# # Map ticks to ax1
# y2_min, y2_max = ax2.get_ylim()
# y1_min, y1_max = ax1.get_ylim()

# y1_ticks = y1_min + (y2_ticks - y2_min) * (y1_max - y1_min) / (y2_max - y2_min)

# ax2.yaxis.set_major_locator(FixedLocator(y2_ticks))
# ax1.yaxis.set_major_locator(FixedLocator(y1_ticks))

# Transform y2 ticks to display space, then to ax1 data space
disp = ax2.transData.transform(np.column_stack([np.zeros_like(y2_ticks), y2_ticks]))
y1_ticks = ax1.transData.inverted().transform(disp)[:, 1]

ax1.yaxis.set_major_locator(FixedLocator(y1_ticks))
ax2.yaxis.set_major_locator(FixedLocator(y2_ticks))



# Set colors to the lines, labels and spines
ax1.lines[0].set_color(color_ax1)
ax2.lines[0].set_color(color_ax2)
ax1.yaxis.label.set_color(color_ax1)
ax2.yaxis.label.set_color(color_ax2)
ax1.spines["left"].set_color(color_ax1)
ax2.spines["right"].set_color(color_ax2)

for label in ax1.get_yticklabels():
    label.set_color(color_ax1)

for label in ax2.get_yticklabels():
    label.set_color(color_ax2)


def sci_1_decimal(x, pos):
    # Display in scientific notation with 1 decimal place
    if x == 0:
        return "0"
    else:
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = x / 10**exponent
        return r"${:.2f} \times 10^{{{}}}$".format(coeff, exponent)

ax1.yaxis.set_major_formatter(FuncFormatter(sci_1_decimal))

# Setup minor ticks
ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
ax2.tick_params(which="minor", length=4)

# Align minor ticks on both axes
y2_minor = ax2.yaxis.get_minorticklocs()

# Transform ax2 data → display → ax1 data
disp = ax2.transData.transform(
    np.column_stack([np.zeros_like(y2_minor), y2_minor])
)
y1_minor = ax1.transData.inverted().transform(disp)[:, 1]

ax1.yaxis.set_minor_locator(FixedLocator(y1_minor))
ax1.tick_params(which="minor", length=4)


# Remove minor ticks from the x axis (both axes share x)
ax1.xaxis.set_minor_locator(mpl.ticker.NullLocator())
ax2.xaxis.set_minor_locator(mpl.ticker.NullLocator())


# Do not show the upper spine
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Avoid double grid
ax2.grid(False)

# Save figure as eps
plt.savefig(os.path.join(ROOT, 'results/ablation-inner-iters/plot_ablation_inner_iters.eps'), format='eps')
plt.savefig(os.path.join(ROOT, 'results/ablation-inner-iters/plot_ablation_inner_iters.png'), format='png', dpi=300)
plt.show()