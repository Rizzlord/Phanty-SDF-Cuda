from context import *

def load_objs(data_dir="data/large-scale-test"):
    """
    Load all .obj files directly under 'data_dir' (non-recursive).

    Returns
    -------
    list of tuples
        Each tuple is (name, V, F), where 'name' is the filename.
    """
    data_dir = Path(data_dir)
    objs = []

    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() != ".obj" or not path.is_file():
            continue

        name = path.name
        print(bcolors.OKGREEN + f"Loading {name}" + bcolors.ENDC)

        try:
            V, F = gpy.read_mesh(str(path))
            if V.size == 0 or F.size == 0:
                print(bcolors.FAIL + f"  Skipping {name}: empty mesh" + bcolors.ENDC)
                continue

            V = gpy.normalize_points(V) * 1.5
            F = utility.triangulate_quads(V, F)
            # Remove .obj from the name
            prefix = name[:-4]
            objs.append((prefix, V, F))
            print(bcolors.OKGREEN + f"  Loaded {name}: {V.shape[0]} vertices, {F.shape[0]} faces" + bcolors.ENDC)
        except Exception as e:
            print(bcolors.FAIL + f"  Failed to load {path}: {e}" + bcolors.ENDC)

    return objs


def compute_all_metrics(V_pred, F_pred, V_gt, F_gt, U, S, rng, n_samples=5000000):
    chamfer = utility.compute_chamfer_dist(V_pred, F_pred, V_gt, F_gt, rng=rng, n_samples=n_samples)
    hdff = utility.compute_hausdorff_distance(V_pred, F_pred, V_gt, F_gt, rng=rng, n_samples=n_samples)
    ecd_L2, ecd_L1, ef1 = utility.compute_edge_chamfer_dist(V_pred, F_pred, V_gt, F_gt)
    sdf_energy = utility.compute_sdf_energy(V_pred, F_pred, U, S)
    return chamfer, hdff, ecd_L2, ecd_L1, ef1, sdf_energy


def run_large_scale_test(max_num_shapes=20, out_csv=None, rng=None, seed=42, res=50,
                         visualize=False, n_samples=5000000):
    out_path = Path(out_csv)
    out_stream = open(out_path, "w")

    out_stream.write("shape,res,method,chamfer,hdff,ecd_L2,ecd_L1,ef1,sdf_energy,time_sec,n_verts\n")
    out_stream.flush()  # Write header immediately

    opts = {
        "dc": {"method": contouring._contouring_cpp_module.ContouringMethod.DualContouring},
        "mc": {"method": contouring._contouring_cpp_module.ContouringMethod.MarchingCubes},
        "ours": {
            "method": contouring._contouring_cpp_module.ContouringMethod.Ours,
            "outer_iters": 100,
            "inner_iters": 100,
            "hermite_update": True,
            "new_hermite_pos_weight": 0.2,
            "new_face_pos_weight": 0.2,
            "new_hermite_normal_weight": 0.2,
            "mu": 0.1,
            "dc_weight": 0.02,
            "verbose": True,
            "batch_size": 200000,
        }
    }

    objs = load_objs()
    if len(objs) == 0:
        print(bcolors.FAIL + "No OBJ files found under data/ - aborting test." + bcolors.ENDC)
        out_stream.close()
        return

    # ps.init()
    all_results = []

    # Cap the number of shapes to max_num_shapes
    count = min(max_num_shapes, len(objs))
    for idx in tqdm(range(count)):
        name, V_gt, F_gt = objs[idx]
        shape_tests = []
        mesh_list = []

        print("\n" + bcolors.OKGREEN + f"Processing {name} ({idx+1}/{count})..." + bcolors.ENDC)
        sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]
        U = contouring.build_grid((res, res, res))
        S = sdf(U)
        
        print(bcolors.OKGREEN + f"  Running Marching Cubes..." + bcolors.ENDC)
        t0 = time.time()
        # safe defaults in case the method fails
        V_mc = np.zeros((0, 3))
        F_mc = np.zeros((0, 3), dtype=int)
        try:
            V_mc, F_mc = contouring.py_contouring(S, U, res, res, res, 0.0, opts['mc'], None, None)
            t_mc = time.time() - t0
            F_mc = utility.triangulate_quads(V_mc, F_mc)
            chamfer_mc, hd_mc, ecd_L2_mc, ecd_L1_mc, ef1_mc, sdf_energy_mc = compute_all_metrics(V_mc, F_mc, V_gt, F_gt, U, S, rng, n_samples=n_samples)
            n_verts = V_mc.shape[0]
            shape_tests.append(("MC", chamfer_mc, hd_mc, ecd_L2_mc, ecd_L1_mc, ef1_mc, sdf_energy_mc, t_mc, n_verts))
            mesh_list.append((V_mc, F_mc, f"mc"))
            print(f"    MC: chamfer={chamfer_mc:.6e}, hdff={hd_mc:.6e}, ecd_L2={ecd_L2_mc:.6e}, ecd_L1={ecd_L1_mc:.6e}, ef1={ef1_mc:.6e}, sdf_energy={sdf_energy_mc:.6e}, time={t_mc:.4f}s, n_verts={n_verts}")
        except Exception as e:
            t_mc = time.time() - t0
            chamfer_mc = hd_mc = ecd_L2_mc = ecd_L1_mc = ef1_mc = sdf_energy_mc = float('nan')
            n_verts = 0
            shape_tests.append(("MC", chamfer_mc, hd_mc, ecd_L2_mc, ecd_L1_mc, ef1_mc, sdf_energy_mc, t_mc, n_verts))
            print(bcolors.FAIL + f"    MC failed: {e}" + bcolors.ENDC)

        print(bcolors.OKGREEN + f"  Running Dual Contouring..." + bcolors.ENDC)
        t0 = time.time()
        # safe defaults in case the method fails
        V_dc = np.zeros((0, 3))
        F_dc = np.zeros((0, 3), dtype=int)
        try:
            V_dc, F_dc = contouring.py_contouring(S, U, res, res, res, 0.0, opts['dc'], None, None)
            t_dc = time.time() - t0
            F_dc = utility.triangulate_quads(V_dc, F_dc)
            chamfer_dc, hd_dc, ecd_L2_dc, ecd_L1_dc, ef1_dc, sdf_energy_dc = compute_all_metrics(V_dc, F_dc, V_gt, F_gt, U, S, rng, n_samples=n_samples)
            n_verts = V_dc.shape[0]
            shape_tests.append(("DC", chamfer_dc, hd_dc, ecd_L2_dc, ecd_L1_dc, ef1_dc, sdf_energy_dc, t_dc, n_verts))
            mesh_list.append((V_dc, F_dc, f"dc"))
            print(f"    DC: chamfer={chamfer_dc:.6e}, hdff={hd_dc:.6e}, ecd_L2={ecd_L2_dc:.6e}, ecd_L1={ecd_L1_dc:.6e}, ef1={ef1_dc:.6e}, sdf_energy={sdf_energy_dc:.6e}, time={t_dc:.4f}s, n_verts={n_verts}")
        except Exception as e:
            t_dc = time.time() - t0
            chamfer_dc = hd_dc = ecd_L2_dc = ecd_L1_dc = ef1_dc = sdf_energy_dc = float('nan')
            n_verts = 0
            shape_tests.append(("DC", chamfer_dc, hd_dc, ecd_L2_dc, ecd_L1_dc, ef1_dc, sdf_energy_dc, t_dc, n_verts))
            print(bcolors.FAIL + f"    DC failed: {e}" + bcolors.ENDC)

        # RFTA 
        if res <= 100:
            print(bcolors.OKGREEN + f"  Running RFTA..." + bcolors.ENDC)
            t0 = time.time()
            V_rfta = np.zeros((0, 3))
            F_rfta = np.zeros((0, 3), dtype=int)
            try:
                V_rfta, F_rfta = gpy.reach_for_the_arcs(U, S, rng_seed=seed)
                t_rfta = time.time() - t0
                F_rfta = utility.triangulate_quads(V_rfta, F_rfta)
                chamfer_rfta, hd_rfta, ecd_L2_rfta, ecd_L1_rfta, ef1_rfta, sdf_energy_rfta = compute_all_metrics(V_rfta, F_rfta, V_gt, F_gt, U, S, rng, n_samples=n_samples)
                n_verts = V_rfta.shape[0]
                shape_tests.append(("RFTA", chamfer_rfta, hd_rfta, ecd_L2_rfta, ecd_L1_rfta, ef1_rfta, sdf_energy_rfta, t_rfta, n_verts))
                mesh_list.append((V_rfta, F_rfta, f"rfta"))
                print(bcolors.OKGREEN + f"    RFTA: chamfer={chamfer_rfta:.6e}, hdff={hd_rfta:.6e}, ecd_L2={ecd_L2_rfta:.6e}, ecd_L1={ecd_L1_rfta:.6e}, ef1={ef1_rfta:.6e}, sdf_energy={sdf_energy_rfta:.6e}, time={t_rfta:.4f}s, n_verts={n_verts}" + bcolors.ENDC)
            except Exception as e:
                t_rfta = time.time() - t0
                chamfer_rfta = hd_rfta = ecd_L2_rfta = ecd_L1_rfta = ef1_rfta = sdf_energy_rfta = float('nan')
                n_verts = 0
                shape_tests.append(("RFTA", chamfer_rfta, hd_rfta, ecd_L2_rfta, ecd_L1_rfta, ef1_rfta, sdf_energy_rfta, t_rfta, n_verts))
                print(bcolors.FAIL + f"    RFTA failed: {e}" + bcolors.ENDC)

        # MNM1 (Kohlbrenner / cones)
        if res <= 100:
            print(bcolors.OKGREEN + f"  Running MNM1..." + bcolors.ENDC)
            t0 = time.time()
            V_mnm1 = np.zeros((0, 3))
            F_mnm1 = np.zeros((0, 3), dtype=int)
            try:
                V_mnm1, F_mnm1 = utility.kohlbrenner_reconstruction(U, S, method='cones')
                t_mnm1 = time.time() - t0
                F_mnm1 = utility.triangulate_quads(V_mnm1, F_mnm1)
                chamfer_mnm1, hd_mnm1, ecd_L2_mnm1, ecd_L1_mnm1, ef1_mnm1, sdf_energy_mnm1 = compute_all_metrics(V_mnm1, F_mnm1, V_gt, F_gt, U, S, rng, n_samples=n_samples)
                n_verts = V_mnm1.shape[0]
                shape_tests.append(("MNM1", chamfer_mnm1, hd_mnm1, ecd_L2_mnm1, ecd_L1_mnm1, ef1_mnm1, sdf_energy_mnm1, t_mnm1, n_verts))
                mesh_list.append((V_mnm1, F_mnm1, f"mnm1"))
                print(bcolors.OKGREEN + f"    MNM1: chamfer={chamfer_mnm1:.6e}, hdff={hd_mnm1:.6e}, ecd_L2={ecd_L2_mnm1:.6e}, ecd_L1={ecd_L1_mnm1:.6e}, ef1={ef1_mnm1:.6e}, sdf_energy={sdf_energy_mnm1:.6e}, time={t_mnm1:.4f}s, n_verts={n_verts}" + bcolors.ENDC)
            except Exception as e:
                t_mnm1 = time.time() - t0
                chamfer_mnm1 = hd_mnm1 = ecd_L2_mnm1 = ecd_L1_mnm1 = ef1_mnm1 = sdf_energy_mnm1 = float('nan')
                n_verts = 0
                shape_tests.append(("MNM1", chamfer_mnm1, hd_mnm1, ecd_L2_mnm1, ecd_L1_mnm1, ef1_mnm1, sdf_energy_mnm1, t_mnm1, n_verts))
                print(bcolors.FAIL + f"    MNM1 failed: {e}" + bcolors.ENDC)
            
        print(bcolors.OKGREEN + f"  Running MNM2..." + bcolors.ENDC)
        t0 = time.time()
        V_mnm2 = np.zeros((0, 3))
        F_mnm2 = np.zeros((0, 3), dtype=int)
        try:
            V_mnm2, F_mnm2 = utility.kohlbrenner_reconstruction(U, S, method='RC', V_gt=V_gt, F_gt=F_gt)
            t_mnm2 = time.time() - t0
            F_mnm2 = utility.triangulate_quads(V_mnm2, F_mnm2)
            chamfer_mnm2, hd_mnm2, ecd_L2_mnm2, ecd_L1_mnm2, ef1_mnm2, sdf_energy_mnm2 = compute_all_metrics(V_mnm2, F_mnm2, V_gt, F_gt, U, S, rng, n_samples=n_samples)
            n_verts = V_mnm2.shape[0]
            shape_tests.append(("MNM2", chamfer_mnm2, hd_mnm2, ecd_L2_mnm2, ecd_L1_mnm2, ef1_mnm2, sdf_energy_mnm2, t_mnm2, n_verts))
            mesh_list.append((V_mnm2, F_mnm2, f"mnm2"))
            print(bcolors.OKGREEN + f"    MNM2: chamfer={chamfer_mnm2:.6e}, hdff={hd_mnm2:.6e}, ecd_L2={ecd_L2_mnm2:.6e}, ecd_L1={ecd_L1_mnm2:.6e}, ef1={ef1_mnm2:.6e}, sdf_energy={sdf_energy_mnm2:.6e}, time={t_mnm2:.4f}s, n_verts={n_verts}" + bcolors.ENDC)
        except Exception as e:
            t_mnm2 = time.time() - t0
            chamfer_mnm2 = hd_mnm2 = ecd_L2_mnm2 = ecd_L1_mnm2 = ef1_mnm2 = sdf_energy_mnm2 = float('nan')
            n_verts = 0
            shape_tests.append(("MNM2", chamfer_mnm2, hd_mnm2, ecd_L2_mnm2, ecd_L1_mnm2, ef1_mnm2, sdf_energy_mnm2, t_mnm2, n_verts))
            print(bcolors.FAIL + f"    MNM2 failed: {e}" + bcolors.ENDC)

        # Run ours
        print(bcolors.OKGREEN + f"  Running Ours..." + bcolors.ENDC)
        t0 = time.time()
        V_ours = np.zeros((0, 3))
        F_ours = np.zeros((0, 3), dtype=int)
        try:
            V_ours, F_ours = contouring.py_contouring(S, U, res, res, res, 0.0, opts['ours'], None, None)
            t_ours = time.time() - t0
            F_ours = utility.triangulate_quads(V_ours, F_ours)
            chamfer_ours, hd_ours, ecd_L2_ours, ecd_L1_ours, ef1_ours, sdf_energy_ours = compute_all_metrics(V_ours, F_ours, V_gt, F_gt, U, S, rng, n_samples=n_samples)
            n_verts = V_ours.shape[0]
            shape_tests.append(("Ours", chamfer_ours, hd_ours, ecd_L2_ours, ecd_L1_ours, ef1_ours, sdf_energy_ours, t_ours, n_verts))
            mesh_list.append((V_ours, F_ours, f"ours"))
            print(bcolors.OKGREEN + f"    Ours: chamfer={chamfer_ours:.6e}, hdff={hd_ours:.6e}, ecd_L2={ecd_L2_ours:.6e}, ecd_L1={ecd_L1_ours:.6e}, ef1={ef1_ours:.6e}, sdf_energy={sdf_energy_ours:.6e}, time={t_ours:.4f}s, n_verts={n_verts}" + bcolors.ENDC)
        except Exception as e:
            t_ours = time.time() - t0
            chamfer_ours = hd_ours = ecd_L2_ours = ecd_L1_ours = ef1_ours = sdf_energy_ours = float('nan')
            n_verts = 0
            shape_tests.append(("Ours", chamfer_ours, hd_ours, ecd_L2_ours, ecd_L1_ours, ef1_ours, sdf_energy_ours, t_ours, n_verts))
            print(bcolors.FAIL + f"    Ours failed: {e}" + bcolors.ENDC)

        for method_name, chamf, hd, ecd_L2, ecd_L1, ef1, sdf_energy, t, n_verts in shape_tests:
            row = {
                'shape': name,
				'res': res,
                'method': method_name,
                'chamfer': chamf,
				'hdff': hd,
				'ecd_L2': ecd_L2,
				'ecd_L1': ecd_L1,
				'ef1': ef1,
				'sdf_energy': sdf_energy,
				'time_sec': t,
                'n_verts': n_verts,
            }
            all_results.append(row)
            out_stream.write(f"{row['shape']},{row['res']},{row['method']},{row['chamfer']:.10f},{row['hdff']:.10f},{row['ecd_L2']:.10f},{row['ecd_L1']:.10f},{row['ef1']:.10f},{row['sdf_energy']:.10f},{row['time_sec']:.10f},{row['n_verts']}\n")
            out_stream.flush()


        # Take screenshots
        # screenshots_dir = ROOT / 'results' / 'large-scale-test' / 'screenshots'
        # for V_mesh, F_mesh, shortname in mesh_list:
        #     mesh_name = f"{name}_res{res}_{shortname}"
        #     fname = screenshots_dir / f"{mesh_name}.png"
        #     ps.remove_all_structures()
        #     ps.register_surface_mesh(shortname, V_mesh, F_mesh)
        #     ps.screenshot(str(fname), 800, 600)


        if visualize:
            ps.remove_all_structures()
            for V_mesh, F_mesh, shortname in mesh_list:
                ps.register_surface_mesh(shortname, V_mesh, F_mesh)
            ps.show()

        # Save the objs
        objs_dir = ROOT / 'results' / 'large-scale-test'
        for V_mesh, F_mesh, shortname in mesh_list:
            mesh_name = f"{name}_res{res}_{shortname}"
            fname = objs_dir / f"{mesh_name}.obj"
            gpy.write_mesh(str(fname), V_mesh, F_mesh)

    out_stream.close()

    # Print aggregated summary per method
    if len(all_results) == 0:
        print(bcolors.WARNING + "No results to summarize." + bcolors.ENDC)
        return
    
    print("\n" + bcolors.HEADER + "Aggregated Results Summary:" + bcolors.ENDC)

	# Compute per-method averages
    methods = sorted(set(row['method'] for row in all_results))
    avg = {}

    for method in methods:
        rows = [r for r in all_results if r['method'] == method]
        n = len(rows)

        avg[method] = {
			'chamfer':      sum(r['chamfer'] for r in rows) / n,
			'hdff':         sum(r['hdff'] for r in rows) / n,
			'ecd_L2':       sum(r['ecd_L2'] for r in rows) / n,
			'ecd_L1':       sum(r['ecd_L1'] for r in rows) / n,
			'ef1':          sum(r['ef1'] for r in rows) / n,
			'sdf_energy':   sum(r['sdf_energy'] for r in rows) / n,
			'time_sec':     sum(r['time_sec'] for r in rows) / n,
            'n_verts':      sum(r['n_verts'] for r in rows) / n,
		}

	# Find the best (lowest) method per metric
    metrics = ['chamfer', 'hdff', 'ecd_L2', 'ecd_L1', 'ef1', 'sdf_energy', 'time_sec', 'n_verts']
    best_method = {
		m: min(methods, key=lambda meth: avg[meth][m])
		for m in metrics
	}

	# Print per-method results
    for method in methods:
        print(bcolors.OKBLUE + f" Method: {method}" + bcolors.ENDC)

        for metric, label in [
			('chamfer',    "Average Chamfer Distance"),
			('hdff',       "Average Hausdorff Distance"),
			('ecd_L2',     "Average Edge Chamfer L2"),
			('ecd_L1',     "Average Edge Chamfer L1"),
			('ef1',        "Average Edge F1 Score"),
			('sdf_energy', "Average SDF Energy"),
            ('time_sec',   "Average Time (sec)"),
            ('n_verts',    "Average Number of Vertices"),
		]:
            value = avg[method][metric]
            color = bcolors.OKCYAN if best_method[metric] == method else ""
            endc  = bcolors.ENDC if color else ""
            print(f"   {color}{label}: {value:.6e}{endc}")

        print(f"   Average Time (sec): {avg[method]['time_sec']:.4f}\n")


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # out = str(ROOT / 'results' / 'large-scale-test' / 'results_50_rfta.csv')
    # run_large_scale_test(max_num_shapes=50, out_csv=out, rng=rng, seed=seed, res=50)

    # out = str(ROOT / 'results' / 'large-scale-test' / 'results_100_rfta.csv')
    # run_large_scale_test(max_num_shapes=50, out_csv=out, rng=rng, seed=seed, res=100)

    # out = str(ROOT / 'results' / 'large-scale-test' / 'results_150_rfta.csv')
    # run_large_scale_test(max_num_shapes=50, out_csv=out, rng=rng, seed=seed, res=150)

    out = str(ROOT / 'results' / 'large-scale-test' / 'results_200.csv')
    run_large_scale_test(max_num_shapes=70, out_csv=out, rng=rng, seed=seed, res=200)