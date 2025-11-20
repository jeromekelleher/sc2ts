import sc2ts

ds = sc2ts.Dataset("../viridian_mafft_2024-10-14_v1.vcz.zip")
print(ds)

samples = ds["sample_id"][:]
k = 1000
samples = samples[:k]
path = f"viridian_mafft_subset_{k}_v1.vcz"
ds.copy(path, sample_id=samples)
sc2ts.Dataset.create_zip(path, path + ".zip")

