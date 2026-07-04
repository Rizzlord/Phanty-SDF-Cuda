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
prefix = "chichen-itza_pyramid"


V_gt, _, _, F_gt, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-uw" / f"{prefix}_gt.obj"))
V_dc, _, _, F_dc, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-uw" / f"{prefix}_dc.obj"))
V_mnm1, _, _, F_mnm1, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-uw" / f"{prefix}_mnm1.obj"))
V_mnm2, _, _, F_mnm2, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-uw" / f"{prefix}_mnm2.obj"))
V_rfta, _, _, F_rfta, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-uw" / f"{prefix}_rfta.obj"))

# Triangulate all
F_dc = utility.triangulate_quads(V_dc, F_dc)
F_mnm1 = utility.triangulate_quads(V_mnm1, F_mnm1)
F_mnm2 = utility.triangulate_quads(V_mnm2, F_mnm2)
F_rfta = utility.triangulate_quads(V_rfta, F_rfta)


# outer_iters_list = [0, 1, 10, 20, 50, 100, 200]
outer_iters_list = [1, 10, 20, 50, 100, 200]
uw_list = [0.01, 0.5, 1.0]

V_ours = {}
F_ours = {}
hdff_dists_ours = {}
chamfer_dists_ours = {}

for outer_iters in outer_iters_list:
    for uw in uw_list:
        V, _, _, F, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-uw" / f"{prefix}_uw{uw}_ot{outer_iters}_ours.obj"))
        V_ours[outer_iters, uw] = V
        F_ours[outer_iters, uw] = F

        # Triangulate faces if needed
        F = utility.triangulate_quads(V, F)

        chamfer_dist = utility.compute_chamfer_dist(V, F, V_gt, F_gt, rng=rng, n_samples=n_samples)
        hdff_dist = utility.compute_hausdorff_distance(V, F, V_gt, F_gt, rng=rng)
        if uw not in chamfer_dists_ours:
            chamfer_dists_ours[uw] = []
            hdff_dists_ours[uw] = []
        chamfer_dists_ours[uw].append(chamfer_dist)
        hdff_dists_ours[uw].append(hdff_dist)
        print(f"Ours uw={uw} outer_iters={outer_iters}: Chamfer distance = {chamfer_dist:.6e}, Hausdorff distance = {hdff_dist:.6e}")

lines = [
    "Ours",
    "dc",
    "mnm1",
    "mnm2",
    "rfta"
]

# Show the meshes in polyscope
# ps.init()
# for method in lines:
#     if method.lower() == "ours":
#         for outer_iters in mu_list:
#             ps.register_surface_mesh(f"ours_outer_iters_{outer_iters}", V_ours[outer_iters], F_ours[outer_iters])
#     elif method.lower() == "dc":
#         ps.register_surface_mesh("dc", V_dc, F_dc)
#     elif method.lower() == "mnm1":
#         ps.register_surface_mesh("mnm1", V_mnm1, F_mnm1)
#     elif method.lower() == "mnm2":
#         ps.register_surface_mesh("mnm2", V_mnm2, F_mnm2)
#     elif method.lower() == "rfta":
#         ps.register_surface_mesh("rfta", V_rfta, F_rfta)
# ps.register_surface_mesh("gt", V_gt, F_gt)
# ps.show()

chamfer_dist_dc = utility.compute_chamfer_dist(V_dc, F_dc, V_gt, F_gt, rng=rng, n_samples=n_samples)
# hdff_dist_dc = utility.compute_hausdorff_distance(V_dc, F_dc, V_gt, F_gt, rng=rng)
chamfer_dist_mnm1 = utility.compute_chamfer_dist(V_mnm1, F_mnm1, V_gt, F_gt, rng=rng, n_samples=n_samples)
# hdff_dist_mnm1 = utility.compute_hausdorff_distance(V_mnm1, F_mnm1, V_gt, F_gt, rng=rng)
chamfer_dist_mnm2 = utility.compute_chamfer_dist(V_mnm2, F_mnm2, V_gt, F_gt, rng=rng, n_samples=n_samples)
# hdff_dist_mnm2 = utility.compute_hausdorff_distance(V_mnm2, F_mnm2, V_gt, F_gt, rng=rng)
chamfer_dist_rfta = utility.compute_chamfer_dist(V_rfta, F_rfta, V_gt, F_gt, rng=rng, n_samples=n_samples)
# hdff_dist_rfta = utility.compute_hausdorff_distance(V_rfta, F_rfta, V_gt, F_gt, rng=rng)


fig, ax = plt.subplots(figsize=(10, 6))

type_to_color = definitions.type_to_color
linestyles = {
    0.01: "dashed",
    0.1: "dashed",
    0.5: "solid",
    1.0: "dashdot",
}

colors = {
    0.01: definitions.ours_color2,
    0.1: definitions.ours_color2,
    0.5: definitions.ours_color,
    1.0: definitions.ours_color3,
}

for uw in uw_list:
    color = colors[uw]

    ax.plot(
        outer_iters_list,
        chamfer_dists_ours[uw],
        marker="o",
        linestyle=linestyles[uw],
        color=color,
        label=rf"$uw={uw}$",
    )

ax.set_xlabel("Outer iterations")
ax.set_ylabel("Chamfer distance")

# Tick formatting
ax.yaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(lambda y, _: f"{y:.3e}")
)

# Disable minor ticks
ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

# Spine styling
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False)
ax.set_ylim(bottom=0)
# Do not show legend
ax.legend().set_visible(False)
# Make y log
#ax.set_yscale("log")

# Save figure as eps
plt.savefig(os.path.join(ROOT, 'results/ablation-uw/plot_ablation_uw.eps'), format='eps')
plt.savefig(os.path.join(ROOT, 'results/ablation-uw/plot_ablation_uw.png'), format='png', dpi=300)
plt.show()