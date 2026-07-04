from context import *


###################################################################################################
################################# Compute errors and save to CSV ##################################
###################################################################################################

rng = np.random.default_rng(42)
prefix = "Rathalos_Head_Low_Poly"

df = pd.read_csv(os.path.join(ROOT, 'results/ours-vs-rfta-error-and-runtime/Rathalos_Head_Low_Poly_timings.csv'))

V_gt, _, _, F_gt, _, _ = igl.readOBJ(str(ROOT / "results" / "ours-vs-rfta-error-and-runtime" / f"{prefix}_gt.obj"))

hdff_dists = []
chamfer_dists = []

for index, row in df.iterrows():
    gs = row['gs']
    method = row['method']
    batch_size = row['batch_size']

    V, _, _, F, _, _ = igl.readOBJ(str(ROOT / "results" / "ours-vs-rfta-error-and-runtime" / f"{prefix}_gs{gs}_bs{batch_size}_{method.lower()}.obj"))
    # Triangulate faces if needed
    F = utility.triangulate_quads(V, F)
    chamfer_ours_orig = utility.compute_chamfer_dist(V, F, V_gt, F_gt, rng=rng)
    hd_ours_orig = utility.compute_hausdorff_distance(V, F, V_gt, F_gt, rng=rng)
    chamfer_dists.append(chamfer_ours_orig)
    hdff_dists.append(hd_ours_orig)

    print(f"Computed distances for gs={gs}, method={method}, batch_size={batch_size}")
    print(f"  Chamfer Distance: {chamfer_ours_orig}")
    print(f"  Hausdorff Distance: {hd_ours_orig}")

df['hdff_distance'] = hdff_dists
df['chamfer_distance'] = chamfer_dists

df.to_csv(os.path.join(ROOT, 'results/ours-vs-rfta-error-and-runtime/Rathalos_Head_Low_Poly_timings_and_errors.csv'), index=False)



###################################################################################################
################################### Plot errors and runtimes ######################################
###################################################################################################



alpha_small = 0.45   # Smaller batch size
alpha_large = 0.85   # Larger batch size

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

df = pd.read_csv(os.path.join(ROOT, 'results/ours-vs-rfta-error-and-runtime/Rathalos_Head_Low_Poly_timings_and_errors.csv'))
df["num_spheres"] = df["gs"] ** 3     # Number of spheres is grid size cubed

# Strip whitespace from column names
df.columns = df.columns.str.strip()
df = df.rename(columns={"time(s)": "time"})

print(df.head())    

lines = [
    ("RFTA", 10000),
    ("RFTA", 100000),
    ("Ours", 100000),
    # ("Ours", 1000000),
]

def batch_to_str(bs):
    if bs >= 1_000_000:
        return f"{bs//1_000_000}M"
    return f"{bs//1_000}K"

# fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for method, batch in lines:
    sub = df[
        (df["method"] == method) &
        (df["batch_size"] == batch)
    ].sort_values("num_spheres")

    type_to_color = definitions.type_to_color
    color = type_to_color.get(method.lower(), definitions.grey_1)

    # Select the alpha based on the batch size
    batches_for_method = df[df["method"] == method]["batch_size"].unique()
    min_batch = batches_for_method.min()

    alpha = alpha_large if batch == min_batch else alpha_small

    axes[0].plot(
        sub["num_spheres"],
        sub["time"],
        marker="o",
        color=color,
        alpha=alpha,
        label=f"{method}, batch={batch}"
    )

    axes[1].plot(
        sub["num_spheres"],
        sub["chamfer_distance"],
        marker="o",
        color=color,
        alpha=alpha,
        label=f"{method}, batch={batch}"
    )

    # axes[2].plot(
    #     sub["num_spheres"],
    #     sub["hdff_distance"],
    #     marker="o",
    #     color=color,
    #     alpha=alpha,
    #     label=f"{method}, batch={batch}"
    # )

    x_last = sub["num_spheres"].iloc[-1]
    y_last = sub["time"].iloc[-1]

    label = f"{method.lower()} (bs={batch_to_str(batch)})"
    axes[0].text(x_last, y_last, label, fontsize=10, color=color, alpha=alpha)


axes[0].set_ylabel('Runtime (s)')
axes[1].set_ylabel('Chamfer Distance')
# axes[2].set_ylabel('Hausdorff Distance')

# Cut y axis at 0.99
axes[1].set_ylim(bottom=0.0007, top=0.2)


for ax in axes:
    ax.set_xlabel('Number of spheres')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Do not show minor ticks
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    # Do not show the upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Increase the spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Save figure as eps
plt.savefig(os.path.join(ROOT, 'results/ours-vs-rfta-error-and-runtime/ours_vs_rfta_error_and_runtime.eps'), format='eps')
plt.savefig(os.path.join(ROOT, 'results/ours-vs-rfta-error-and-runtime/ours_vs_rfta_error_and_runtime.png'), format='png', dpi=300)
plt.show()