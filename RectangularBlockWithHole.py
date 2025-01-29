# Importing necessary modules
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Function to create a mesh with a circular hole in the center
def create_mesh_with_hole(x, y, nx, ny, radius, num_hole_nodes):
    # Dictionary to store node coordinates
    node_coordinates = {}

    # Create a rectangular grid of nodes
    x_coords = np.linspace(0, x, nx)
    y_coords = np.linspace(0, y, ny)
    xv, yv = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([xv.ravel(), yv.ravel()])

    # Define the center of the circular hole
    circle_center = (x / 2, y / 2)

    # Remove nodes that lie inside the circular hole
    distances = np.sqrt((points[:, 0] - circle_center[0])**2 + (points[:, 1] - circle_center[1])**2)
    outside_points = points[distances >= radius]

    # Generate nodes around the circular hole boundary
    theta = np.linspace(0, 2 * np.pi, num_hole_nodes, endpoint=False)
    circle_points = np.column_stack([
        circle_center[0] + radius * np.cos(theta),
        circle_center[1] + radius * np.sin(theta)
    ])

    # Combine the rectangular and circular boundary nodes
    total_points = np.vstack([outside_points, circle_points])
    for i, (px, py) in enumerate(total_points):
        node_name = f'R{i}'
        node_coordinates[node_name] = (px, py)

    # Connect circular boundary nodes to the nearest outer node to form boundary triangles
    boundary_triangles = []
    for i in range(num_hole_nodes):
        p1 = len(outside_points) + i
        p2 = len(outside_points) + ((i + 1) % num_hole_nodes)

        edge_midpoint = (total_points[p1] + total_points[p2]) / 2.0
        distances_to_edge_midpoint = np.sqrt(np.sum((outside_points - edge_midpoint)**2, axis=1))
        nearest_point_idx = np.argmin(distances_to_edge_midpoint)
        boundary_triangles.append([p1, p2, nearest_point_idx])

    # Perform Delaunay triangulation of all points
    delaunay = Delaunay(total_points)
    triangles = delaunay.simplices

    # Filter out triangles whose centroids fall inside the circular hole
    triangle_centroids = np.mean(total_points[triangles], axis=1)
    triangle_distances = np.sqrt((triangle_centroids[:, 0] - circle_center[0])**2 + 
                                 (triangle_centroids[:, 1] - circle_center[1])**2)

    exterior_triangles = []
    for triangle, centroid_distance in zip(triangles, triangle_distances):
        if centroid_distance >= radius:
            exterior_triangles.append(triangle)

    # Combine boundary and exterior triangles and remove duplicates
    TEMP_TRI = boundary_triangles + exterior_triangles
    unique_triangles = set(tuple(sorted(tri)) for tri in TEMP_TRI)
    all_triangles = [list(tri) for tri in unique_triangles]

    # Display the triangular elements with their node IDs
    print("Triangles formed by nodes:")
    for tri in sorted(all_triangles):
        R1, R2, R3 = tri
        print(f"R{R1}, R{R2}, R{R3}")

    return node_coordinates, all_triangles

# Remove any existing global stiffness matrix files
os.system("rm global_k*.txt")

# Elastic modulus and Poisson's ratio
E = 210e3
nu = 0.3

# Function to calculate shape function and its derivatives
def shape_function(x1, y1, x2, y2, x3, y3):
    N = np.zeros((3,3))
    B = np.zeros((3,6))
    for i in range(0,3):
        A = np.array([[1, x1, y1],
                      [1, x2, y2],
                      [1, x3, y3]])
        b = np.array([0, 0, 0])
        b[i] = 1
        N[i,:] = np.linalg.solve(A, b)
        B[0, 2*i]   = N[i, 1]
        B[1, 2*i+1] = N[i, 2]
        B[2, 2*i]   = N[i, 2]
        B[2, 2*i+1] = N[i, 1]
        element_area = abs(0.5 * np.linalg.det(A))
    print("Shape functions (B):\n", B)
    print("Element area:", element_area)
    return B, element_area

# Function to compute the elasticity matrix
def elasticity_matrix(E, nu):
    preFactor = E / (1 - nu**2)
    C = preFactor * np.array([[1, nu, 0],
                               [nu, 1, 0],
                               [0,  0, (1 - nu)/2]])
    return C

# Function to compute the stiffness matrix for an element
def el_stiffness_matrix(x1, y1, x2, y2, x3, y3):
    B, element_area = shape_function(x1, y1, x2, y2, x3, y3)
    C = elasticity_matrix(E, nu)
    K_elem = element_area * B.T @ C @ B
    # Print elemental stiffness matrix and its determinant
    print("Elemental stiffness matrix (K_elem):\n", K_elem)
    print("Determinant of K_elem:", np.linalg.det(K_elem))

    return K_elem

# Map element nodes to their global degrees of freedom
def get_global_dof(node_IDs):
    dofs = []
    for node in node_IDs:
        dofs.extend([2 * node, 2 * node + 1])
    return dofs

# Function to assemble the global stiffness matrix
def global_stiffness_matrix(node_coordinates, elements):
    nodal_dof = 2
    K_global = np.zeros((nodal_dof * len(node_coordinates), nodal_dof * len(node_coordinates)))
    for element in elements:
        one, two, three = element
        x1, y1 = node_coordinates["R" + str(one)]
        x2, y2 = node_coordinates["R" + str(two)]
        x3, y3 = node_coordinates["R" + str(three)]

        # Compute the element stiffness matrix
        K_elem = el_stiffness_matrix(x1, y1, x2, y2, x3, y3)

        # Get global DOF indices
        gDOF = get_global_dof(element)

        # Assemble into the global stiffness matrix
        for i in range(len(K_elem)):
            for j in range(len(K_elem)):
                K_global[gDOF[i], gDOF[j]] += K_elem[i, j]

    return K_global

# Function to apply boundary conditions (essential and natural)
def apply_boundary_conditions(K_global, F_global, ebc, nbc):
    # Apply essential boundary conditions
    for bc1 in ebc:
        node, u, v = bc1['node'], bc1['u'], bc1['v']
        if u is not None:
            gid_u = 2 * node
            K_global[gid_u, :] = 0.0
            K_global[:, gid_u] = 0.0
            K_global[gid_u, gid_u] = 1.0
            F_global[gid_u] = u

        if v is not None:
            gid_v = 2 * node + 1
            K_global[gid_v, :] = 0.0
            K_global[:, gid_v] = 0.0
            K_global[gid_v, gid_v] = 1.0
            F_global[gid_v] = v

    # Apply natural boundary conditions (forces)
    for bc2 in nbc:
        node, Fx, Fy = bc2['node'], bc2['Fx'], bc2['Fy']
        gid_Fx, gid_Fy = 2 * node, 2 * node + 1
        if Fx is not None:
            F_global[gid_Fx] += Fx
        if Fy is not None:
            F_global[gid_Fy] += Fy

# Solve the linear system to obtain displacements
def matrix_solve(K_global, F_global):
    K_sparse = csr_matrix(K_global)
    displacements = spsolve(K_sparse, F_global)
    return displacements

# Define boundary conditions for the problem
def define_boundary_conditions(node_coordinates, y_max, total_force_y):
    ebc = []
    nbc = []

    top_boundary_nodes = []
    # Fix nodes at the bottom boundary
    for node_name, (px, py) in node_coordinates.items():
        node_id = int(node_name[1:])
        if np.isclose(py, 0.0):
            ebc.append({'node': node_id, 'u': 0.0, 'v': 0.0})
        elif np.isclose(py, y_max):
            top_boundary_nodes.append(node_id)

    # Distribute force along the top boundary nodes
    num_top_nodes = len(top_boundary_nodes)
    force_per_node_y = total_force_y / num_top_nodes if num_top_nodes > 0 else 0.0

    for node_id in top_boundary_nodes:
        nbc.append({'node': node_id, 'Fx': 0.0, 'Fy': force_per_node_y})

    return ebc, nbc

# Function to plot the mesh and node numbers
def plot_mesh_with_node_numbers(node_coordinates, elements):
    node_ids = sorted(node_coordinates.keys(), key=lambda x: int(x[1:]))
    coords = np.array([node_coordinates[nid] for nid in node_ids])

    x_vals = coords[:, 0]
    y_vals = coords[:, 1]

    plt.figure(figsize=(8, 8))
    plt.triplot(x_vals, y_vals, elements, color='black', linewidth=0.7)
    for i, nid in enumerate(node_ids):
        px, py = node_coordinates[nid]
        plt.plot(px, py, 'ro', markersize=0.1)
        plt.text(px, py, nid, color='red', fontsize=8, ha='center', va='center')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Mesh with Node Numbers")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Problem setup and initialization
x = 10
y = 10
nx = 10
ny = 10
radius = 4
num_hole_nodes = 8

# Generate mesh and nodes
node_coordinates, elements = create_mesh_with_hole(x, y, nx, ny, radius, num_hole_nodes)

# Prepare global stiffness matrix and force vector
nodal_dof = 2
K_global = np.zeros((nodal_dof * len(node_coordinates), nodal_dof * len(node_coordinates)))
F_global = np.zeros(nodal_dof * len(node_coordinates))

# Build global stiffness matrix
K_global = global_stiffness_matrix(node_coordinates, elements)
np.savetxt("global_k.txt", K_global, fmt="  %.5f")

y_max = max(node[1] for node in node_coordinates.values())
total_force_y = 1000.0

# Define and display boundary conditions
ebc, nbc = define_boundary_conditions(node_coordinates, y_max, total_force_y)

print("Essential Boundary Conditions (EBC):")
for bc in ebc:
    print(bc)

print("\nNatural Boundary Conditions (NBC):")
for bc in nbc:
    print(bc)

# Apply boundary conditions
apply_boundary_conditions(K_global, F_global, ebc, nbc)

# Save the modified stiffness matrix and force vector
np.savetxt("global_k_ebc.txt", K_global, fmt="  %.5f")
np.savetxt("global_f_ebc.txt", F_global, fmt="  %.5f")

# Solve for displacements
displacements = matrix_solve(K_global, F_global)

if displacements is not None:
    u = np.zeros(len(node_coordinates))
    v = np.zeros(len(node_coordinates))

    # Display final displacement values
    print("\nNode Displacements:")
    for node_id, (px, py) in node_coordinates.items():
        idx = int(node_id[1:])
        u_val = displacements[2 * idx]
        v_val = displacements[2 * idx + 1]
        u[idx] = u_val
        v[idx] = v_val
        print(f"{node_id}: u = {u_val:.6f}, v = {v_val:.6f}")

    # Verify the solution
    print("Solution check:", np.allclose(K_global @ displacements, F_global))

    # Plot the mesh with node numbers
    plot_mesh_with_node_numbers(node_coordinates, elements)

    # Create a grid for displacement visualization
    xs, ys = np.meshgrid(np.linspace(min(node_coordinates.values(), key=lambda x: x[0])[0],
                                     max(node_coordinates.values(), key=lambda x: x[0])[0], 200),
                         np.linspace(min(node_coordinates.values(), key=lambda x: x[1])[1],
                                     max(node_coordinates.values(), key=lambda x: x[1])[1], 200))

    # Interpolate displacement fields for visualization
    us = griddata(points=[(px, py) for px, py in node_coordinates.values()], values=u, xi=(xs, ys), method='linear')
    vs = griddata(points=[(px, py) for px, py in node_coordinates.values()], values=v, xi=(xs, ys), method='linear')

    # Plot U displacement
    plt.figure(figsize=(4, 3))
    h = plt.contourf(xs, ys, us, cmap='jet')
    cbar = plt.colorbar(h)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('U displacement')
    plt.plot([px for px, py in node_coordinates.values()],
             [py for px, py in node_coordinates.values()], 'ro', markersize=4)
    plt.show()

    # Plot V displacement
    plt.figure(figsize=(4, 3))
    h = plt.contourf(xs, ys, vs, cmap='jet')
    cbar = plt.colorbar(h)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('V displacement')
    plt.plot([px for px, py in node_coordinates.values()],
             [py for px, py in node_coordinates.values()], 'ro', markersize=4)
    plt.show()

else:
    print("Solver did not return a displacement solution.")
