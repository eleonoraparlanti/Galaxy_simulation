import numpy as np

def generate_grid_in_box(box_left_edge,box_right_edge,N_grid):
	"""
	It generates the grid centers in the simulation box in all 3 directoins.
	Output : x_grid_center, y_grid_center, z_grid_center
	"""
	dx = (box_right_edge-box_left_edge)/N_grid
	#return np.linspace(0+dx/2,L_box-dx/2,N_grid),np.linspace(0+dx/2,L_box-dx/2,N_grid),np.linspace(0+dx/2,L_box-dx/2,N_grid)
	return np.linspace(box_left_edge+dx/2,box_right_edge-dx/2,N_grid)
	
def projection(box_left_edge,box_right_edge,N_grid,pos_array,field_array,normalization=1.0):
	"""
	This function project the given quantity onto the grid (mesh) points
	"""
	n_dim, N_pt = np.shape(pos_array)
	dx = (box_right_edge-box_left_edge)/N_grid 
	print("dx = ",dx)
	grid_centers = generate_grid_in_box(box_left_edge,box_right_edge,N_grid)	
	projected_field      = np.zeros((N_grid+2,N_grid+2,N_grid+2))
	
	for i in range(N_pt):
		
		referGrid            = np.zeros(n_dim)
		weightScalarGridDown = np.zeros(n_dim)
		weightScalarGridUp   = np.zeros(n_dim)
		for j in range(n_dim):
		
			try:
				#referGrid[j]        = np.where(abs(pos_array[j][i]-grid_centers)<dx)[0][0]
				referGrid[j]        = np.where(((pos_array[j][i]-grid_centers)>0)&((pos_array[j][i]-grid_centers)<dx))[0][0]
				broke = False
			except IndexError:
				broke = True
				break
			
			weightScalarGridUp[j]   = (pos_array[j,i] - grid_centers[int(referGrid[j])])/dx
			weightScalarGridDown[j]	= 1 - weightScalarGridUp[j]
		
		if(broke):
			continue
			
		projected_field[int(referGrid[0])][int(referGrid[1])][int(referGrid[2])]       += field_array[i] * weightScalarGridDown[0] * weightScalarGridDown[1] * weightScalarGridDown[2]
		projected_field[int(referGrid[0])][int(referGrid[1])+1][int(referGrid[2])]     += field_array[i] * weightScalarGridDown[0] * weightScalarGridUp[1]   * weightScalarGridDown[2]
		projected_field[int(referGrid[0])][int(referGrid[1])][int(referGrid[2])+1]     += field_array[i] * weightScalarGridDown[0] * weightScalarGridDown[1] * weightScalarGridUp[2]
		projected_field[int(referGrid[0])][int(referGrid[1])+1][int(referGrid[2])+1]   += field_array[i] * weightScalarGridDown[0] * weightScalarGridUp[1]   * weightScalarGridUp[2]
		projected_field[int(referGrid[0])+1][int(referGrid[1])][int(referGrid[2])]     += field_array[i] * weightScalarGridUp[0]   * weightScalarGridDown[1] * weightScalarGridDown[2]
		projected_field[int(referGrid[0])+1][int(referGrid[1])+1][int(referGrid[2])]   += field_array[i] * weightScalarGridUp[0]   * weightScalarGridUp[1]   * weightScalarGridDown[2]
		projected_field[int(referGrid[0])+1][int(referGrid[1])][int(referGrid[2])+1]   += field_array[i] * weightScalarGridUp[0]   * weightScalarGridDown[1] * weightScalarGridUp[2]
		projected_field[int(referGrid[0])+1][int(referGrid[1])+1][int(referGrid[2])+1] += field_array[i] * weightScalarGridUp[0]   * weightScalarGridUp[1]   * weightScalarGridUp[2]
		
	return projected_field
	
################################
### FOR THE TEST THIS FUNCRION
################################
"""
box_left_edge   = 0
box_right_edge  = 10
N_grid = 10
dx = (box_right_edge-box_left_edge)/N_grid
grid_centers = generate_grid_in_box(box_left_edge,box_right_edge,N_grid)
#pos_array    = (dx/2) + (L_box-dx) * np.random.random((3,10)) 
#pos_array    =  (box_right_edge-box_left_edge) * np.random.random((3,10))
pos_array    = np.array((15.3,7.3,3.2,5.3,7.3,3.2,5.3,7.3,3.2))
pos_array    = pos_array.reshape(3,3)
mass_array   = np.ones(10)
print("pos = ",pos_array)
time_start = time()
out = projection(box_left_edge,box_right_edge,N_grid,pos_array,mass_array)
print(out)
print("time for the projection = ",time()-time_start)
print("sum of all mass = ",np.sum(out))
h = h5py.File('data_new.h5', 'w')
dset = h.create_dataset('density', data=out)

if((weightScalarGridDown[j]<0) or (weightScalarGridUp[j]<0)):
				print("pos = ",pos_array[:,i])
				print("box left side = ",box_left_edge, "    box right side = ",box_right_edge)
				print("dx = ",dx)
				#print("difference = ",abs(pos_array[j][i]-grid_centers))
				print("box left side = ",box_left_edge-dx, "    box right side = ",box_right_edge+dx)
				print( " for i = ",i," and j =",j, "  refere grid centers = ",grid_centers[int(referGrid[j])], " weighted scalar = ",weightScalarGridDown[j],weightScalarGridUp[j])
				print("referGrid = ",referGrid)
				print("\n")
		
"""
