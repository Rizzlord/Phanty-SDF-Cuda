blue_1 = (166.0/255.0,206.0/255.0,227.0/255.0,1.0)   #a6cee3
blue_2 = (31.0/255.0,120.0/255.0,180.0/255.0,1.0)    #1f78b4
green_1 = (178.0/255.0,223.0/255.0,138.0/255.0,1.0)  #b2df8a
green_2 = (51.0/255.0,160.0/255.0,44.0/255.0,1.0)
red_1 = (251.0/255.0,154.0/255.0,153.0/255.0,1.0)
red_2 = (227.0/255.0,26.0/255.0,28.0/255.0,1.0)
orange_1 = (253.0/255.0,191.0/255.0,111.0/255.0,1.0)
orange_2 = (255.0/255.0,127.0/255.0,0.0/255.0,1.0)
purple_1 = (202.0/255.0,178.0/255.0,214.0/255.0,1.0)
purple_2 = (106.0/255.0,61.0/255.0,154.0/255.0,1.0)
yellow_1 = (255.0/255.0,255.0/255.0,153.0/255.0,1.0)
yellow_2 = (225.0/255.0,214.0/255.0,35.0/255.0,1.0)
brown = (177.0/255.0,89.0/255.0,40.0/255.0,1.0)
grey_1 = (128.0/255.0,128.0/255.0,128.0/255.0,1.0)
grey_2 = (45.0/255.0,45.0/255.0,45.0/255.0,1.0)
pink = (221/255.0,52/255.0,151/255.0,1.0)    #DD3497


input_mesh_color = grey_1
positive_sphere_color = blue_2
negative_sphere_color = red_2
generic_sphere_color = grey_1

# These are placeholders, we should tweak them later
rfts_color = green_2
rfta_color = blue_2
marching_cubes_color = orange_2
neural_dual_contouring_color = purple_2
dual_contouring_color = purple_2
mnm1_color = green_2
mnm2_color = yellow_2
ours_color = pink

# Mapping from result type string to display color
type_to_color = {
	'gt': input_mesh_color,
	'ours': ours_color,
	'mc': marching_cubes_color,
	'ndc': neural_dual_contouring_color,
    'dc': dual_contouring_color,
    'mnm1': mnm1_color,
    'mnm2': mnm2_color,
	'rfts': rfts_color,
	'rfta': rfta_color,
}