#include <time.h>
#include <stdlib.h>

#include <ParticleFilter.cuh>
#include <CudaUtils.cuh>

#include <common.cuh>

namespace SLAM{

__global__ void normal_init_kernel(curandState *state, float seed, int len){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= len) return;

    curand_init(seed, idx, 0, &state[idx]);     /* Each thread gets different seed, a different sequence number, no offset */
}

__global__ void set_fvalue_kernel(float *array, float value, int len){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= len) return;

    array[idx] = value;
}

__global__ void set_ivalue_kernel(int *array, int value, int len){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= len) return;

    array[idx] = value;
}

__global__ void sum_kernel(float *array_1, float *array_2, int len){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= len) return;

    array_1[idx] += array_2[idx];
}

__global__ void ranges_threshold_kernel(float *ranges, valid_t *ranges_valid ,float min, float max ,int len){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= len) return;

    if(ranges[idx] < min ) ranges_valid[idx] = VIVL;
    else if(ranges[idx] > max) {
        ranges_valid[idx] = VINF;
        ranges[idx] = max;
    }
    else ranges_valid[idx] = VVLD;
}

__global__ void base_polar_to_cart_local_kernel(float *ranges, float *angles, valid_t *valids, int beam_count,
                                                float* thetas, int particle_count,
                                                int *cart_x , int *cart_y,
                                                float m_per_cell, int len){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= len) return;

    int range_idx = idx % beam_count;
    int theta_idx = idx / beam_count;

    if(valids[range_idx] == VIVL) return;

    float range = ranges[range_idx];
    float angle = angles[range_idx];
    float theta = thetas[theta_idx];

    float x = range*std::cos(angle+theta);
    float y = range*std::sin(angle+theta);

    cart_x[idx] = round(x/m_per_cell);
    cart_y[idx] = round(y/m_per_cell);
}

__global__ void scan_base_polar_to_cart_kernel( float *dev_scan_x , float *dev_scan_y,                  // output
                                                float *thetas, float *ranges, float *angles,            // input
                                                valid_t *valids, int beam_count,                        // input
                                                int len)                                                 // valid check
{                                               

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= len) return;

    int range_idx = idx % beam_count;
    int theta_idx = idx / beam_count;

    if(valids[range_idx] == VIVL) return;

    float range = ranges[range_idx];
    float angle = angles[range_idx];
    float theta = thetas[theta_idx];

    dev_scan_x[idx] = range*std::cos(angle+theta);
    dev_scan_y[idx] = range*std::sin(angle+theta);
}

__global__ void scan_base_to_map_kernel(float *dev_scan_map_x, float *dev_scan_map_y,                          // ouput
                                        float *dev_scan_base_x, float *dev_scan_base_y, valid_t *scan_valids,       // input
                                        float *dev_base_map_x, float *dev_base_map_y, int beam_count,           // input
                                        int len){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= len) return;
    if(scan_valids[idx%beam_count] == VIVL) return;

    int particle_idx = idx / beam_count;

    dev_scan_map_x[idx] = dev_scan_base_x[idx] + dev_base_map_x[particle_idx];
    dev_scan_map_y[idx] = dev_scan_base_y[idx] + dev_base_map_y[particle_idx];
}

__global__ void find_correspondence_kernel( float *dev_corr_map_x, float *dev_corr_map_y, valid_t *dev_corr_valid,
                                            float *dev_scan_map_x, float *dev_scan_map_y, valid_t *dev_scan_valid, float m_per_cell,
                                            float *global_map, int global_width_cell, int global_height_cell, int global_size_cell,
                                            int beam_count, int kernel_size,
                                            uint32_t len){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;

    // Reset coresspondance
    dev_corr_map_x[idx] = 0;
    dev_corr_map_y[idx] = 0;
    dev_corr_valid[idx] = VIVL;

    if(dev_scan_valid[idx%beam_count] == VIVL) return;
                                            
    float scan_x = dev_scan_map_x[idx];
    float scan_y = dev_scan_map_y[idx];

    int particle_idx = idx / beam_count;
    global_map += particle_idx * global_size_cell;

    int stop_x = global_width_cell/2 + round(scan_x / m_per_cell);
    int stop_y = global_height_cell/2 + round(scan_y / m_per_cell);
    if(stop_x >= global_width_cell || stop_y >= global_height_cell) return;

    float lkh, min_dis=1.0f;
    for(int dx=-kernel_size; dx<=kernel_size; dx++){
        for(int dy=-kernel_size; dy<=kernel_size; dy++){
            int x = stop_x + dx;
            int y = stop_y + dy;
            int cell = y * global_width_cell + x;
            if(cell < 0 || cell >= global_size_cell) continue;
            lkh  = global_map[cell];
            if(lkh > 0){
                float x_m = (x - global_width_cell/2)*m_per_cell;
                float y_m = (y - global_height_cell/2)*m_per_cell;
                float dis = (x_m - scan_x)*(x_m - scan_x) + (y_m-scan_y)*(y_m-scan_y);
                if(dis < min_dis){
                    min_dis = dis;
                    dev_corr_map_x[idx] = x_m;
                    dev_corr_map_y[idx] = y_m;
                    dev_corr_valid[idx] = VVLD;
                }
            }
        }
    }
}

#define CHECK(x,y)      if(x>=map_width || y>=map_height || x<0 || y<0) return;
#define MAP_IDX(x,y)    map[y*map_width + x]
#define FREE(x,y)       {CHECK(x,y); MAP_IDX(x,y) -= free;}
#define OCCUPY(x,y)     {CHECK(x,y); MAP_IDX(x,y) += occ;}
#define OCCUPY2(x,y)    {CHECK(x,y); MAP_IDX(x,y) += free + occ;}
__global__ void bresenham_occupancy_update_kernel(float *map, int map_width, int map_height,
                                                    int start_x, int start_y,  int beam_count,
                                                    int *stop_x_ptr, int *stop_y_ptr, valid_t *valid,
                                                    float occ, float free, int len){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= len) return;

    int particle_idx = idx/beam_count;
    map += particle_idx*map_height*map_width;

    idx = idx%beam_count;
    
    if(valid[idx] == VIVL) return;

    start_x += map_width/2;
    start_y += map_height/2;

    int stop_x = start_x + stop_x_ptr[idx];
    int stop_y = start_y + stop_y_ptr[idx];

    int dx = abs(start_x - stop_x); 
    int dy = abs(start_y - stop_y);

    int x = start_x;
    int y = start_y;

    if(dx > dy){
        int inc1 = 2*dy;
        int inc2 = 2*(dy-dx);
        int d = 2*dy - dx;

        FREE(x,y);
        if(start_x < stop_x){
            if(start_y < stop_y){
                while (x < stop_x) {
                    x++;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        y++; d+=inc2;
                    }
                    FREE(x,y);
                }
            }
            else{
                while (x < stop_x) {
                    x++;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        y--; d+=inc2;
                    }
                    FREE(x,y);
                }
            }
        }
        else{
            if(start_y < stop_y){
                while (x > stop_x) {
                    x--;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        y++; d+=inc2;
                    }
                    FREE(x,y);
                }
            }
            else{
                while (x > stop_x) {
                    x--;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        y--; d+=inc2;
                    }
                    FREE(x,y);
                }
            }
        }
    }
    else{
        int inc1 = 2*dx;
        int inc2 = 2*(dx-dy);
        int d = 2*dx - dy;

        FREE(x,y);
        if(start_y < stop_y){
            if(start_x < stop_x){
                while (y < stop_y) {
                    y++;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        x++; d+=inc2;
                    }
                    FREE(x,y);
                }
            }
            else{
                while (y < stop_y) {
                    y++;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        x--; d+=inc2;
                    }
                    FREE(x,y);
                }
            }
        }
        else{
            if(start_x < stop_x){
                while (y > stop_y) {
                    y--;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        x++; d+=inc2;
                    }
                    FREE(x,y);
                }
            }
            else{
                while (y > stop_y) {
                    y--;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        x--; d+=inc2;
                    }
                    FREE(x,y);
                }
            }
        }
    }

    if(valid[idx] == VVLD) OCCUPY2(x,y);
}

__global__ void add_local_map_to_global_map_kernel(float *global_map, int global_map_width_cell, int global_map_height_cell,
                                                    float *local_map, int local_map_width_cell, int local_map_height_cell,
                                                    float *local_origin_x_m, float *local_origin_y_m,
                                                    float m_per_cell, int local_size_cell, int global_size_cell,
                                                    float *dev_w, float w_thresh,
                                                    int particle_count, int max_x, int max_y){
    int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    if(idx_x>=max_x || idx_y>=max_y) return;

    int particle_idx = idx_x / local_map_width_cell;
    if(dev_w[particle_idx] < w_thresh) return;
    
    global_map += particle_idx*global_size_cell;
    local_map += particle_idx*local_size_cell;

    int x_cell = idx_x%local_map_width_cell;
    int y_cell = idx_y;

    int local_x_cell = x_cell - local_map_width_cell/2;
    int local_y_cell = y_cell - local_map_height_cell/2;

    int local_origin_x_cell = round(local_origin_x_m[particle_idx] / m_per_cell);
    int local_origin_y_cell = round(local_origin_y_m[particle_idx] / m_per_cell);

    int global_x_cell = local_origin_x_cell + local_x_cell;
    int global_y_cell = local_origin_y_cell + local_y_cell;

    if(abs(global_x_cell) >= global_map_width_cell/2 || abs(global_y_cell) >= global_map_height_cell/2) return;

    global_x_cell += global_map_width_cell/2;
    global_y_cell += global_map_height_cell/2;

    float occupancy = global_map[global_y_cell*global_map_width_cell + global_x_cell];
    float additional_occupancy = local_map[y_cell*local_map_width_cell + x_cell];
    
    if((occupancy < -5 && additional_occupancy < 0) || occupancy > 5 && additional_occupancy > 0) return;

    global_map[global_y_cell*global_map_width_cell + global_x_cell] = occupancy + additional_occupancy;
}

#define MAP(x,y)    global_map[(y) * global_width_cell + (x)];
#define LKH(x,y)    int cell = y*global_width_cell+x;   \
                    if( cell < 0 || cell >= global_size_cell) continue; \
                    float lkh = global_map[cell];
                    
__global__ void likelihood_kernel(  float *likelihood,  valid_t *likehood_valids,
                                    float *global_map, float m_per_cell, valid_t *scan_valids,
                                    int global_width_cell, int global_height_cell, int global_size_cell,
                                    float *local_origin_x_m, float *local_origin_y_m, int particle_count, int beam_count,
                                    float *dev_scan_map_x, float *dev_scan_map_y, int len){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;

    likelihood[idx] = 0;
    likehood_valids[idx] = VIVL;

    if(scan_valids[idx%beam_count] == VIVL) return;

    int particle_idx = idx / beam_count;
    global_map += particle_idx*global_size_cell;

    int start_x = global_width_cell/2 + round(local_origin_x_m[particle_idx] / m_per_cell);
    int start_y = global_height_cell/2 + round(local_origin_y_m[particle_idx] / m_per_cell);
    
    int stop_x = global_width_cell/2 + round(dev_scan_map_x[idx] / m_per_cell);
    int stop_y = global_height_cell/2 + round(dev_scan_map_y[idx] / m_per_cell);

    if(start_x >= global_width_cell || start_y >= global_height_cell || start_x < 0 || start_y < 0) return;
    if(stop_x >= global_width_cell || stop_y >= global_height_cell || stop_x < 0 || stop_y < 0) return;

    int dx = abs(start_x - stop_x); 
    int dy = abs(start_y - stop_y);

    int x = start_x;
    int y = start_y;

    if(dx > dy){
        int inc1 = 2*dy;
        int inc2 = 2*(dy-dx);
        int d = 2*dy - dx;

        if(start_x < stop_x){
            if(start_y < stop_y){
                while (x < stop_x) {
                    x++;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        y++; d+=inc2;
                    }
                    LKH(x,y);
                    if(lkh > 0){
                        float dist = (x-stop_x)*(x-stop_x) + (y-stop_y)*(y-stop_y);
                        likelihood[idx] = lkh/exp(dist);
                        break;
                    }
                }
            }
            else{
                while (x < stop_x) {
                    x++;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        y--; d+=inc2;
                    }
                    LKH(x,y);
                    if(lkh > 0){
                        float dist = (x-stop_x)*(x-stop_x) + (y-stop_y)*(y-stop_y);
                        likelihood[idx] = lkh/exp(dist);
                        break;
                    }
                }
            }
        }
        else{
            if(start_y < stop_y){
                while (x > stop_x) {
                    x--;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        y++; d+=inc2;
                    }
                    LKH(x,y);
                    if(lkh > 0){
                        float dist = (x-stop_x)*(x-stop_x) + (y-stop_y)*(y-stop_y);
                        likelihood[idx] = lkh/exp(dist);
                        break;
                    } 
                }
            }
            else{
                while (x > stop_x) {
                    x--;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        y--; d+=inc2;
                    }
                    LKH(x,y);
                    if(lkh > 0){
                        float dist = (x-stop_x)*(x-stop_x) + (y-stop_y)*(y-stop_y);
                        likelihood[idx] = lkh/exp(dist);
                        break;
                    }
                }
            }
        }
    }
    else{
        int inc1 = 2*dx;
        int inc2 = 2*(dx-dy);
        int d = 2*dx - dy;

        if(start_y < stop_y){
            if(start_x < stop_x){
                while (y < stop_y) {
                    y++;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        x++; d+=inc2;
                    }
                    LKH(x,y);
                    if(lkh > 0){
                        float dist = (x-stop_x)*(x-stop_x) + (y-stop_y)*(y-stop_y);
                        likelihood[idx] = lkh/exp(dist);
                        break;
                    }
                }
            }
            else{
                while (y < stop_y) {
                    y++;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        x--; d+=inc2;
                    }
                    LKH(x,y);
                    if(lkh > 0){
                        float dist = (x-stop_x)*(x-stop_x) + (y-stop_y)*(y-stop_y);
                        likelihood[idx] = lkh/exp(dist);
                        break;
                    }
                }
            }
        }
        else{
            if(start_x < stop_x){
                while (y > stop_y) {
                    y--;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        x++; d+=inc2;
                    }
                    LKH(x,y);
                    if(lkh > 0){
                        float dist = (x-stop_x)*(x-stop_x) + (y-stop_y)*(y-stop_y);
                        likelihood[idx] = lkh/exp(dist);
                        break;
                    }
                }
            }
            else{
                while (y > stop_y) {
                    y--;
                    if (d<0) {
                        d+=inc1;
                    } else {
                        x--; d+=inc2;
                    }
                    LKH(x,y);
                    if(lkh > 0){
                        float dist = (x-stop_x)*(x-stop_x) + (y-stop_y)*(y-stop_y);
                        likelihood[idx] = lkh/exp(dist);
                        break;
                    }
                }
            }
        }
    }
}

__global__ void copy_state_kernel(float *new_state, float *old_state, int *S, int particle_count){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > particle_count) return;
    new_state[idx] = old_state[S[idx]];
}

__global__ void copy_map_kernel(float *new_map, float *old_map, uint32_t len){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx > len) return;
    new_map[idx] = old_map[idx];
}

__global__ void prediction_kernel(float *x, float *y, float *theta,
                                    float dx, float dy, float dtheta,
                                    float noise_x, float noise_y, float noise_theta, uint32_t len){
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > len) return;

    curandState state;
    curand_init(clock64(), idx, 0, &state);
    float m_noise[3];
	m_noise[0] = (curand_normal(&state))*noise_x;
	m_noise[1] = (curand_normal(&state))*noise_y;
	m_noise[2] = (curand_normal(&state))*noise_theta;

    x[idx]      += dx + m_noise[0];
	y[idx]      += dy + m_noise[1];
	theta[idx]  += dtheta + m_noise[2];
}

__global__ void apply_transform_kernel(float *x, float *y, float *t,
                                        float *la, float *c, float *s, float *tx, float *ty,
                                        uint32_t len){

    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;

    float tmp_x = x[idx];
    float tmp_y = y[idx];
    float lac = la[idx] * c[idx];
    float las = la[idx] * s[idx];
    x[idx] = lac*tmp_x - las*tmp_y + tx[idx];
    y[idx] = las*tmp_x + lac*tmp_y + ty[idx];
    t[idx] += atan2(s[idx],c[idx]);
}

}
