import os
import numpy as np
import faiss
import h5py

# --- Load dataset from HDF5 ---
print("Loading dataset from 'dataset.h5'...")
with h5py.File("dataset.h5", "r") as hf:
    # Read each dataset (all are shape [n_rows, 34], int)
    standard_data = hf["standardWinningHands"][:]
    chiitoi_data = hf["chiitoiHands"][:]
    kokushi_data = hf["kokushiHands"][:]

# If you want to treat them as a single large dataset:
data = np.concatenate([standard_data, chiitoi_data, kokushi_data], axis=0).astype('float32')
print("Dataset shape:", data.shape)
d = data.shape[1]

# --- Identify chiitoitsu hands ---
# The old logic for detecting chiitoitsu from the combined data:
chiitoitsu_mask = np.array([
    (np.count_nonzero(row) == 7) and np.all(row[row > 0] == 2)
    for row in data
])
nonchiitoitsu_indices = np.where(~chiitoitsu_mask)[0]
data_nonchiitoitsu = data[~chiitoitsu_mask]

print(f"Total hands: {data.shape[0]}")
print(f"Chiitoitsu hands (detected): {np.sum(chiitoitsu_mask)}")
print(f"Non-chiitoitsu hands: {data_nonchiitoitsu.shape[0]}")

# --- Build full FAISS index ---
nlist = 4000
quantizer = faiss.IndexFlatL2(d)
index_full = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
index_full.nprobe = 10

print("Training full index...")
index_full.train(data)
print("Adding data to full index...")
index_full.add(data)
faiss.write_index(index_full, "full_index.faiss")
print("Full index saved to 'full_index.faiss'.")

# --- Build non-chiitoitsu FAISS index ---
if data_nonchiitoitsu.shape[0] > 0:
    quantizer2 = faiss.IndexFlatL2(d)
    index_nonchiitoitsu = faiss.IndexIVFFlat(quantizer2, d, nlist, faiss.METRIC_L2)
    index_nonchiitoitsu.nprobe = 10
    print("Training non-chiitoitsu index...")
    index_nonchiitoitsu.train(data_nonchiitoitsu)
    print("Adding data to non-chiitoitsu index...")
    index_nonchiitoitsu.add(data_nonchiitoitsu)
    faiss.write_index(index_nonchiitoitsu, "nonchiitoitsu_index.faiss")
    print("Non-chiitoitsu index saved to 'nonchiitoitsu_index.faiss'.")
else:
    # Fallback if there is no non-chiitoitsu data
    index_nonchiitoitsu = index_full
    faiss.write_index(index_nonchiitoitsu, "nonchiitoitsu_index.faiss")
    print("No separate non-chiitoitsu data found; using full index.")

# --- Save metadata ---
np.save("chiitoitsu_mask.npy", chiitoitsu_mask)
np.save("nonchiitoitsu_indices.npy", nonchiitoitsu_indices)
np.save("data.npy", data)
np.save("data_nonchiitoitsu.npy", data_nonchiitoitsu)
print("Metadata saved.")
