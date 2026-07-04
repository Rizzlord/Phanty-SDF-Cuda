import sys, os
# add path ../external/maximal-empty-spheres/cgal to sys.path
import gpytoolbox as gpy


def kohlbrenner_reconstruction(U, S, V_gt = None, F_gt = None, temp_file_name = 'temp.obj', method='cones', delaunay=False):
    # method should be 'cones' or 'RC', depending on the SIGGRAPH or Eurographics paper
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../external/maximal-empty-spheres')))
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../external/maximal-empty-spheres')))
    # so this doesn't crashes your code unless you call this function
    if method == 'cones':
        from cgal.EmptySpheresReconstruction import MESReconstruction
        R_cgal = MESReconstruction(U,S,screening_weight=1.0)
        V = R_cgal[0]
        F = R_cgal[1]
    elif method == 'RC':
        # check that V_gt and F_gt are not None
        if V_gt is None or F_gt is None:
            raise ValueError("V_gt and F_gt must be provided for 'RC' method")
        # grid size is cubic root of U shape 0
        grid_size = int(round(U.shape[0] ** (1.0 / 3.0)))
        # write mesh to temp file
        gpy.write_mesh(temp_file_name, V_gt, F_gt)
        output_temp_file = 'output_' + temp_file_name
        # we need to run, relative to this file's directory:
        # ./../../external/sdf-weighted-delaunay/build/sdf-weighted-delaunay temp_file_name grid_size 0 0 output_temp_file.obj
        delaunay_number = 1 if delaunay else 0
        command = os.path.join(os.path.dirname(__file__), '../external/sdf-weighted-delaunay/build/sdf-weighted-delaunay') + f' {temp_file_name} {grid_size} 0 {delaunay_number} {output_temp_file}'
        print("Running command: ", command)
        os.system(command)
        # read output_temp_file.obj
        V, F = gpy.read_mesh(output_temp_file)
        # remove temp files
        os.remove(temp_file_name)
        os.remove(output_temp_file)
    return V, F
