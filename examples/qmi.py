import random
import dwave.cloud
import dwave.inspector


# get solver
print("solver init")
client = dwave.cloud.Client.from_config()
solver = client.get_solver(qpu=True)

# define problem (over first n qubits)
n = 100
linear = {u: random.uniform(-1, 1) for u in sorted(solver.nodes)[:n]}
quadratic = {(u,v): random.uniform(-1, 1) for u,v in solver.undirected_edges
                                          if u in linear and v in linear}
problem = (linear, quadratic)

# sample
print("sampling")
response = solver.sample_ising(*problem, num_reads=100)

# inspect
print("inspecting")
dwave.inspector.show(problem, response)
