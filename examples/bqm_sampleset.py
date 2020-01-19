import dimod
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite


# define problem
bqm = dimod.BQM.from_ising({}, {'ab': 1, 'bc': 1, 'ca': 1})

# get sampler
print("sampler init")
sampler = EmbeddingComposite(DWaveSampler(solver=dict(qpu=True)))

# sample -> sampleset + embedding (+ warnings)
print("sampling")
sampleset = sampler.sample(bqm, num_reads=1)

# inspect
print("inspecting")
dwave.inspector.show_bqm_sampleset(bqm, sampleset, sampler)

# or simply:
# dwave.inspector.show(bqm, sampleset, sampler)
