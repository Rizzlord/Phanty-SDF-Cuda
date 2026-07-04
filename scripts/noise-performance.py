
from context import *


def load_objs():
	"""Return a list of (name, V, F) for .obj files directly under data/ (not recursive)."""
	data_dir = "data/"
	objs = []
	for i, p in enumerate(sorted(Path(data_dir).iterdir())):
		if p.is_file() and p.suffix.lower() == '.obj':
			if i < 4: 
				continue
			name = p.name
			data_name = data_dir + name
			print(f"Loading {name}")
			try:
				V, F = gpy.read_mesh(data_name)
				V = gpy.normalize_points(V)
				objs.append((name, V, F))
				print(f"  Loaded {name}: {V.shape[0]} vertices, {F.shape[0]} faces")
			except Exception as e:
				print(f"Failed to load {name}: {e}")
	return objs


def run_noise_performance_test(num_shapes=20, out_csv=None, rng=None, seed=42, res=50,
							   noise_levels=None, sigma_list=None, visualize=True, n_samples=5000000):
	out_path = Path(out_csv)
	out_stream = open(out_path, "w")

	# CSV header includes noise and sigma
	out_stream.write("name,res,method,noise,sigma,hausdorff,chamfer,time_sec\n")
	out_stream.flush()

	Methods = {
		"DualContouring": contouring._contouring_cpp_module.ContouringMethod.DualContouring,
		"Ours": contouring._contouring_cpp_module.ContouringMethod.Ours,
	}

	if noise_levels is None:
		noise_levels = [0.001, 0.005, 0.01, 0.05]
	if sigma_list is None:
		# Negative sigma disables the denoising term
		sigma_list = [-1.0, 0.0001, 0.001, 0.01, 0.1]

	objs = load_objs()
	if len(objs) == 0:
		print("No OBJ files found under data/ - aborting test.")
		out_stream.close()
		return

	ps.init()
	all_results = []

	count = min(num_shapes, len(objs))
	for idx in tqdm(range(count)):
		name, V_gt, F_gt = objs[idx]

		if V_gt.size == 0 or F_gt.size == 0:
			print(f"Skipping {name}: empty mesh")
			continue

		print(f"\nProcessing {name} ({idx+1}/{count})...")
		sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]
		U = contouring.build_grid((res, res, res))
		S = sdf(U)

		for noise_level in noise_levels:
			# Sample noise once per (shape, noise_level)
			noise = rng.normal(scale=noise_level, size=S.shape)
			S_noisy = S + noise

			tests = []
			
            # Dual Contouring (FD grads)
			opts_dc = {"method": Methods["DualContouring"]}
			print(f"  Running Dual Contouring (noise={noise_level})...")
			t0 = time.time()
			V_dc, F_dc = contouring.py_contouring(S_noisy, U, res, res, res, 0.0, opts_dc, None, None)
			t_dc = time.time() - t0
			F_dc = utility.triangulate_quads(V_dc, F_dc)
			chamfer_dc = utility.compute_chamfer_dist(V_dc, F_dc, V_gt, F_gt, rng=rng, n_samples=n_samples)
			hd_dc = utility.compute_hausdorff_distance(V_dc, F_dc, V_gt, F_gt, rng=rng, n_samples=n_samples)
			tests.append(("DC", hd_dc, chamfer_dc, t_dc))

			# RFTA 
			# print(f"  Running RFTA (noise={noise_level})...")
			# t0 = time.time()
			# V_rfta, F_rfta = gpy.reach_for_the_arcs(U, S_noisy, rng_seed=seed)
			# t_rfta = time.time() - t0
			# F_rfta = utility.triangulate_quads(V_rfta, F_rfta)
			# chamfer_rfta = utility.compute_chamfer_dist(V_rfta, F_rfta, V_gt, F_gt, rng=rng, n_samples=n_samples)
			# hd_rfta = utility.compute_hausdorff_distance(V_rfta, F_rfta, V_gt, F_gt, rng=rng, n_samples=n_samples)
			# tests.append(("RFTA", hd_rfta, chamfer_rfta, t_rfta))

			# MNM1 (Kohlbrenner / cones)
			print(f"  Running MNM1 (noise={noise_level})...")
			t0 = time.time()
			V_mnm1, F_mnm1 = utility.kohlbrenner_reconstruction(U, S_noisy, method='cones')
			t_mnm1 = time.time() - t0
			F_mnm1 = utility.triangulate_quads(V_mnm1, F_mnm1)
			chamfer_mnm1 = utility.compute_chamfer_dist(V_mnm1, F_mnm1, V_gt, F_gt, rng=rng, n_samples=n_samples)
			hd_mnm1 = utility.compute_hausdorff_distance(V_mnm1, F_mnm1, V_gt, F_gt, rng=rng, n_samples=n_samples)
			tests.append(("MNM1", hd_mnm1, chamfer_mnm1, t_mnm1))

			for method_name, hd, chamf, t in tests:
				row = {
					'name': name,
					'res': res,
					'method': method_name,
					'noise': noise_level,
					'sigma': np.nan,
					'hausdorff': hd,
					'chamfer': chamf,
					'time_sec': t,
				}
				all_results.append(row)
				out_stream.write(f"{row['name']},{row['res']},{row['method']},{row['noise']},{row['sigma']},{row['hausdorff']:.10f},{row['chamfer']:.10f},{row['time_sec']:.10f}\n")
				out_stream.flush()
			
			for method_name, hd, chamf, t in tests:
				print(f"  {method_name:10s}: hd={hd:.6e} chamfer={chamf:.6e} time={t:.4f}s")

			tests_ours = []
			V_ours_all = []
			F_ours_all = []
			for sigma in sigma_list:
				# Our method
				opts_ours = {
					"method": Methods["Ours"],
					"outer_iters": 100,
					"inner_iters": 100,
					"hermite_update": True,
					"new_hermite_pos_weight": 0.2,
					"new_face_pos_weight": 0.2,
					"new_hermite_normal_weight": 0.2,
					"mu": 0.1,
					"dc_weight": 0.01,
					"sigma": sigma,
					"verbose": True,
					"batch_size": 200_000,
				}
				print(f"  Running Our Method (noise={noise_level}, sigma={sigma})...")
				t0 = time.time()
				V_ours, F_ours = contouring.py_contouring(S_noisy, U, res, res, res, 0.0, opts_ours, None, None)
				t_ours = time.time() - t0
				F_ours = utility.triangulate_quads(V_ours, F_ours)
				chamfer_ours = utility.compute_chamfer_dist(V_ours, F_ours, V_gt, F_gt, rng=rng, n_samples=n_samples)
				hd_ours = utility.compute_hausdorff_distance(V_ours, F_ours, V_gt, F_gt, rng=rng, n_samples=n_samples)
				tests_ours.append(("Ours", sigma, hd_ours, chamfer_ours, t_ours))
				V_ours_all.append(V_ours)
				F_ours_all.append(F_ours)

				print(f"{name} | res={res} | noise={noise_level} | sigma={sigma} | time={t_ours:.2f}s | hd={hd_ours:.6e} | chamfer={chamfer_ours:.6e}")

			for method_name, sigma, hd, chamf, t in tests_ours:
				print(f"  {method_name:10s} (sigma={sigma}): hd={hd:.6e} chamfer={chamf:.6e} time={t:.4f}s")
				row = {
					'name': name,
					'res': res,
					'method': method_name,
					'noise': noise_level,
					'sigma': sigma,
					'hausdorff': hd,
					'chamfer': chamf,
					'time_sec': t,
				}
				all_results.append(row)
				out_stream.write(f"{row['name']},{row['res']},{row['method']},{row['noise']},{row['sigma']},{row['hausdorff']:.10f},{row['chamfer']:.10f},{row['time_sec']:.10f}\n")
				out_stream.flush()

			
			# Take screenshots       
			screenshots_dir = ROOT / 'results' / 'noise'
			mesh_list = [
				("gt", V_gt, F_gt),
				("DC", V_dc, F_dc),
				*[(f"Ours_sigma{sigma}", V_ours_all[i], F_ours_all[i]) for i, sigma in enumerate(sigma_list)],
				# ("RFTA", V_rfta, F_rfta),
				("MNM1", V_mnm1, F_mnm1),
			]
			for shortname, V_mesh, F_mesh in mesh_list:
				mesh_name = f"{name}_res{res}_noise{noise_level}_{shortname}"
				ps.register_surface_mesh(mesh_name, V_mesh, F_mesh)
				fname = screenshots_dir / f"{mesh_name}.png"
				ps.screenshot(str(fname))
				ps.remove_all_structures()


			if visualize:
				ps.remove_all_structures()
				ps.register_surface_mesh("gt", V_gt, F_gt)
				ps.register_surface_mesh("DC", V_dc, F_dc)
				for i, sigma in enumerate(sigma_list):
					ps.register_surface_mesh(f"Ours_sigma{sigma}", V_ours_all[i], F_ours_all[i])
				# ps.register_surface_mesh("RFTA", V_rfta, F_rfta)
				ps.register_surface_mesh("MNM1", V_mnm1, F_mnm1)
				ps.show()

			# Save the objs
			objs_dir = ROOT / 'results' / 'noise' 
			for shortname, V_mesh, F_mesh in mesh_list:
				mesh_name = f"{name}_res{res}_noise{noise_level}_{shortname}"
				fname = objs_dir / f"{mesh_name}.obj"
				gpy.write_mesh(str(fname), V_mesh, F_mesh)

	out_stream.close()

	# Print aggregated summary per method
	if len(all_results) > 0:
		methods = {}
		for r in all_results:
			m = r['method']
			methods.setdefault(m, []).append(r)

		print('\nSummary (averages over processed runs):')
		print('method\tavg_hausdorff\tavg_chamfer\tavg_time_s\tn')
		for m, rows in methods.items():
			n = len(rows)
			avg_h = sum(rr['hausdorff'] for rr in rows) / n
			avg_c = sum(rr['chamfer'] for rr in rows) / n
			avg_t = sum(rr['time_sec'] for rr in rows) / n
			print(f"{m}\t{avg_h:.6e}\t{avg_c:.6e}\t{avg_t:.4f}\t{n}")


if __name__ == '__main__':
	seed = 42
	np.random.seed(seed)
	rng = np.random.default_rng(seed)

	out = str(ROOT / 'results' / 'noise' / 'results_noise_performance_res50.csv')
	run_noise_performance_test(num_shapes=10, out_csv=out, rng=rng, seed=seed)


