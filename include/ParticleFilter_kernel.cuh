#ifndef __PARTICLE_FILTER_KERNEL__
#define __PARTICLE_FILTER_KERNEL__

#include <stdint.h>

namespace SLAM{

__global__ void normal_init_kernel(curandState *state, float seed, int len);

__global__ void set_fvalue_kernel(float *array, float value, int len);
__global__ void set_ivalue_kernel(int *array, int value, int len);
__global__ void sum_kernel(float *array_1, float *array_2, int len);
__global__ void ranges_threshold_kernel(float *ranges, valid_t *ranges_valid ,float min, float max ,int len);

__global__ void base_polar_to_cart_local_kernel(float *ranges, float *angles, valid_t *valids, int beam_count,
                                                float* thetas, int particle_count,
                                                int *cart_x , int *cart_y,
                                                float m_per_cell, int len);
                                                
__global__ void bresenham_occupancy_update_kernel(float *map, int map_width, int map_height,
                                                    int start_x, int start_y, int beam_count,
                                                    int *stop_x_ptr, int *stop_y_ptr, valid_t *valid,
                                                    float occ, float free, int len);

__global__ void add_local_map_to_global_map_kernel(float *global_map, int global_map_width_cell, int global_map_height_cell,
                                                        float *local_map, int local_map_width_cell, int local_map_height_cell,
                                                        float *local_origin_x_m, float *local_origin_y_m,
                                                        float m_per_cell, int local_size_cell, int global_size_cell,
                                                        float *dev_w, float w_thresh,
                                                        int particle_count, int max_x, int max_y);

__global__ void likelihood_kernel(float *likelihood, valid_t *likehood_valids,
                                float *global_map, float m_per_cell, valid_t *scan_valids,
                                int global_width_cell, int global_height_cell, int global_size_cell,
                                float *local_origin_x_m, float *local_origin_y_m, int particle_count, int beam_count,
                                float *dev_scan_map_x, float *dev_scan_map_y, int len);

__global__ void prediction_kernel(float *x, float *y, float *theta,
                                    float dx, float dy, float dtheta,
                                    float noise_x, float noise_y, float noise_theta, uint32_t len);

__global__ void copy_state_kernel(float *new_state, float *old_state, int *S, int particle_count);
__global__ void copy_map_kernel(float *new_map, float *old_map, uint32_t len);

__global__ void find_correspondence_kernel( float *dev_corr_map_x, float *dev_corr_map_y, valid_t *dev_corr_valid,
                                            float *dev_scan_map_x, float *dev_scan_map_y, valid_t *dev_scan_valid, float m_per_cell,
                                            float *global_map, int global_width_cell, int global_height_cell, int global_size_cell,
                                            int beam_count, int kernel_size,
                                            uint32_t len);

__global__ void scan_base_polar_to_cart_kernel( float *dev_scan_x , float *dev_scan_y,                  // output
                                                float *thetas, float *ranges, float *angles,            // input
                                                valid_t *valids, int beam_count,                        // input
                                                int len);

__global__ void scan_base_to_map_kernel(float *dev_scan_map_x, float *dev_scan_map_y,                           // ouput
                                        float *dev_scan_base_x, float *dev_scan_base_y,  valid_t *scan_valids,  // input
                                        float *dev_base_map_x, float *dev_base_map_y, int beam_count,           // input
                                        int len);

__global__ void apply_transform_kernel(float *x, float *y, float *t,
                                        float *la, float *c, float *s, float *tx, float *ty,
                                        uint32_t len);

}

#endif