
#define MKN(logn)   ((size_t)1 << (logn))


__global__ void sign_tree(fpr *d_t0, fpr *d_t1, fpr *d_b01, fpr *d_b11, const uint16_t *d_hm, int logn,size_t d_mem_pool_pitch);

__global__ void ffSampling_fft_offline(fpr *d_z0, fpr *d_z1, fpr *d_t0, fpr *d_t1, fpr *d_tree, fpr *d_tmp, unsigned orig_logn, unsigned logn, uint64_t *d_scA, uint64_t *d_scdptr, size_t d_mem_pool_pitch);

__global__ void Get_lattice_point(fpr *d_t0, fpr *d_t1, fpr *d_tx, fpr *d_ty, fpr *d_b00, fpr *d_b01, fpr *d_b10, fpr *d_b11, unsigned logn, size_t d_mem_pool_pitch);


