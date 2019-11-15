import sys
import dimod
import dwave.inspector
from minorminer import find_embedding
from dwave.cloud import Client
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.embedding.utils import edgelist_to_adjacency


# define problem
bqm = dimod.BQM.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})

# or, load it from file (if provided)
if len(sys.argv) > 1:
    path = sys.argv[1]
    with open(path) as fp:
        bqm = dimod.BinaryQuadraticModel.from_coo(fp).spin

# get solver
print("solver init")
client = Client.from_config()
solver = client.get_solver(name='DW_2000Q_2_1')

# embed
print("embedding")
source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
target_edgelist = solver.edges
embedding = find_embedding(source_edgelist, target_edgelist)
target_adjacency = edgelist_to_adjacency(target_edgelist)
bqm_embedded = embed_bqm(bqm, embedding, target_adjacency)

# sample
print("sampling")
response = solver.sample_ising(bqm_embedded.linear, bqm_embedded.quadratic, num_reads=100)
sampleset_embedded = response.sampleset
sampleset = unembed_sampleset(sampleset_embedded, embedding, bqm)

# inspect
print("inspecting")
dwave.inspector.show_bqm_response(bqm, embedding, response)
