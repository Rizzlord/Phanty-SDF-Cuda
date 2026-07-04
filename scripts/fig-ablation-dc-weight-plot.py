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
prefix = "great_wall_of_china"


V_gt, _, _, F_gt, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-dc-weight" / f"{prefix}_gt.obj"))
V_dc, _, _, F_dc, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-dc-weight" / f"{prefix}_dc.obj"))
V_mnm1, _, _, F_mnm1, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-dc-weight" / f"{prefix}_mnm1.obj"))
V_mnm2, _, _, F_mnm2, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-dc-weight" / f"{prefix}_mnm2.obj"))
V_rfta, _, _, F_rfta, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-dc-weight" / f"{prefix}_rfta.obj"))

# Triangulate all
F_dc = utility.triangulate_quads(V_dc, F_dc)
F_mnm1 = utility.triangulate_quads(V_mnm1, F_mnm1)
F_mnm2 = utility.triangulate_quads(V_mnm2, F_mnm2)
F_rfta = utility.triangulate_quads(V_rfta, F_rfta)


#dc_weight_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
dc_weight_list = [0.001, 0.01, 0.1, 1.0, 10.0]

V_ours = {}
F_ours = {}
hdff_dists_ours = []
chamfer_dists_ours = []

for dc_weight in dc_weight_list:
    V, _, _, F, _, _ = igl.readOBJ(str(ROOT / "results" / "ablation-dc-weight" / f"{prefix}_dc{dc_weight}_ours.obj"))
    V_ours[dc_weight] = V
    F_ours[dc_weight] = F

    # Triangulate faces if needed
    F = utility.triangulate_quads(V, F)

    chamfer_dist = utility.compute_chamfer_dist(V, F, V_gt, F_gt, rng=rng, n_samples=n_samples)
    hdff_dist = utility.compute_hausdorff_distance(V, F, V_gt, F_gt, rng=rng)
    chamfer_dists_ours.append(chamfer_dist)
    hdff_dists_ours.append(hdff_dist)

    print(f"Ours dc_weight={dc_weight}: Chamfer distance = {chamfer_dist:.6e}, Hausdorff distance = {hdff_dist:.6e}")

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
#         for outer_iters in dc_weight_list:
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

for method in lines:
    type_to_color = definitions.type_to_color
    color = type_to_color.get(method.lower(), definitions.grey_1)

    if method.lower() == "ours":
        label = f"{method}"
        ax.plot(
            dc_weight_list,
            chamfer_dists_ours,
            marker="o",
            color=color,
            label=label
        )
        x_last = dc_weight_list[-1] * 1.01
        y_last = chamfer_dists_ours[-1]
    
    elif method.lower() == "dc":
        label = f"{method} (CD: {chamfer_dist_dc:.2e})"
        ax.hlines(
            chamfer_dist_dc,
            xmin=dc_weight_list[0],
            xmax=dc_weight_list[-1],
            colors=color,
            linestyles="dotted",
            label=label
        )    
        x_last = dc_weight_list[-1] * 0.8
        y_last = chamfer_dist_dc * 1.1

    elif method.lower() == "mnm1":
        label = f"{method} (CD: {chamfer_dist_mnm1:.2e})"
        ax.hlines(
            chamfer_dist_mnm1,
            xmin=dc_weight_list[0],
            xmax=dc_weight_list[-1],
            colors=color,
            linestyles="dashed",
            label=label
        )    
        x_last = dc_weight_list[-1]
        y_last = chamfer_dist_mnm1
    
    elif method.lower() == "mnm2":
        label = f"{method} (CD: {chamfer_dist_mnm2:.2e})"
        ax.hlines(
            chamfer_dist_mnm2,
            xmin=dc_weight_list[0],
            xmax=dc_weight_list[-1],
            colors=color,
            linestyles=(0, (8, 6)),
            label=label
        )    
        x_last = dc_weight_list[-1]
        y_last = chamfer_dist_mnm2

    elif method.lower() == "rfta":
        label = f"{method} (CD: {chamfer_dist_rfta:.2e})"
        ax.hlines(
            chamfer_dist_rfta,
            xmin=dc_weight_list[0],
            xmax=dc_weight_list[-1],
            colors=color,
            linestyles="dashdot",
            label=label
        )    
        x_last = dc_weight_list[-1]
        y_last = chamfer_dist_rfta

    ax.text(x_last, y_last, label, fontsize=10, color=color)

    # Make the y ticks in exponential format
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.3f}'.format(y)))

ax.set_xlabel('Dual contouring weight')
ax.set_ylabel('Chamfer distance')
ax.set_xscale('log')

# Do not show minor ticks
ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

# Do not show the upper and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Make y axis start at zero
ax.set_ylim(bottom=0)


# Save figure as eps
plt.savefig(os.path.join(ROOT, 'results/ablation-dc-weight/plot_ablation_dc_weight.eps'), format='eps')
plt.savefig(os.path.join(ROOT, 'results/ablation-dc-weight/plot_ablation_dc_weight.png'), format='png', dpi=300)
plt.show()