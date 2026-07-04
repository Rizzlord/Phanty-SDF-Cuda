
from context import *

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
prefix = "69079"
results_path = ROOT / "results" / "asymptotic-complexity"
timings_csv_path = results_path / f"{prefix}_timings.csv"

grid_sizes = [10, 25, 50, 75, 100, 125, 150, 200]
grid_sizes_cubed = [gs**3 for gs in grid_sizes]
chamfer_grid_sizes = [125, 200]

if not timings_csv_path.exists():
    raise FileNotFoundError(
        f"Missing timing CSV: {timings_csv_path}. "
        "Run scripts/fig-asymptotic-complexity-model.py first."
    )

timings_df = pd.read_csv(timings_csv_path)
required_columns = {"grid_size", "num_grid_cells", "method", "time_seconds"}
missing_columns = required_columns.difference(timings_df.columns)
if missing_columns:
    raise ValueError(
        f"{timings_csv_path} is missing required columns: "
        f"{sorted(missing_columns)}"
    )

V_gt, _, _, F_gt, _, _ = igl.readOBJ(str(results_path / f"{prefix}_gt.obj"))

chamfer_methods = [
    ("ours", "Ours"),
    ("dc", "Dual Contouring"),
    ("mnm1", "Kohlbrenner Cones"),
    ("rfta", "Reach For The Arcs"),
]

for chamfer_grid_size in chamfer_grid_sizes:
    print(f"Chamfer distances for grid size {chamfer_grid_size}^3:")
    for method_key, method_label in chamfer_methods:
        mesh_path = results_path / f"{prefix}_gs{chamfer_grid_size}_{method_key}.obj"
        if not mesh_path.exists():
            print(f"  - {method_label}: skipped, missing {mesh_path.name}")
            continue

        V, _, _, F, _, _ = igl.readOBJ(str(mesh_path))
        F = utility.triangulate_quads(V, F)
        chamfer_dist = utility.compute_chamfer_dist(V, F, V_gt, F_gt, rng=rng, n_samples=n_samples)
        print(f"  - {method_label}: {chamfer_dist:.6e}")

lines = [
    "Ours",
    "dc",
    "mnm1",
    "rfta"
]


fig, ax = plt.subplots(figsize=(10, 6))

for method in lines:
    type_to_color = definitions.type_to_color
    method_key = method.lower()
    color = type_to_color.get(method_key, definitions.grey_1)
    method_timings = timings_df[timings_df["method"].str.lower() == method_key]
    method_timings = method_timings.sort_values("grid_size")

    if method_timings.empty:
        print(f"Warning: no timings found for method '{method_key}' in {timings_csv_path}")
        continue

    label = method
    ax.plot(
        method_timings["num_grid_cells"],
        method_timings["time_seconds"],
        marker="o",
        color=color,
        label=label
    )

    x_last = method_timings["num_grid_cells"].iloc[-1] * 1.01
    y_last = method_timings["time_seconds"].iloc[-1]

    ax.text(x_last, y_last, label, fontsize=10, color=color)

    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.3f}'.format(y)))

ax.set_xlabel('Grid size (log)')
ax.set_ylabel('Runtime in seconds (log)')
ax.set_xscale('log')
ax.set_yscale('log')

# Do not show minor ticks
ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

# Do not show the upper and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Save figure as eps
# plt.savefig(results_path / 'plot_asymptotic_complexity.eps', format='eps')
# plt.savefig(results_path / 'plot_asymptotic_complexity.png', format='png', dpi=300)
# plt.show()
