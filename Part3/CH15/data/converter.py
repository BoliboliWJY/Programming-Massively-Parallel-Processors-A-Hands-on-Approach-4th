import numpy as np

edges = np.loadtxt("web-Google.txt", dtype=np.int32)
num_vertices = edges.max() + 1
num_edges = edges.shape[0]

def build_csr(src, dst):
    counts = np.bincount(src, minlength=num_vertices)
    ptrs = np.empty(num_vertices + 1, dtype=np.int32)
    ptrs[0] = 0
    np.cumsum(counts, out=ptrs[1:])
    order = np.argsort(src, kind="mergesort")
    return ptrs, dst[order]

src_ptrs, dst = build_csr(edges[:, 0], edges[:, 1])
dst_ptrs, src = build_csr(edges[:, 1], edges[:, 0])

src_ptrs.tofile("srcPtrs.bin")
dst.tofile("dst.bin")
dst_ptrs.tofile("dstPtrs.bin")
src.tofile("src.bin")
print(f"numVertices={num_vertices}, numEdges={num_edges}")
