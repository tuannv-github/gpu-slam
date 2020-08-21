#include <time.h>
#include <stdlib.h>

#include <ParticleFilter.cuh>
#include <ParticleFilter_kernel.cuh>
#include <CudaUtils.cuh>

namespace SLAM{

ParticleFilter::ParticleFilter(int _particle_count) : particle_count(_particle_count){

    int bytes = sizeof(float)*particle_count;

    hst_x       = (float*)malloc(bytes);
    hst_y       = (float*)malloc(bytes);
    hst_t       = (float*)malloc(bytes);
    hst_w       = (float*)malloc(bytes);

    hst_pick       = (int*)malloc(particle_count*sizeof(int));
    CUDA_CALL(cudaMalloc((void**) &dev_pick, particle_count * sizeof(int)));

    CUDA_CALL(cudaMalloc((void**)&dev_x, bytes));
    CUDA_CALL(cudaMalloc((void**)&dev_y, bytes));
    CUDA_CALL(cudaMalloc((void**)&dev_t, bytes));
    CUDA_CALL(cudaMalloc((void**)&dev_w, bytes));

    CUDA_CALL(cudaMalloc((void**)&dev_x_new, particle_count * sizeof(curandState_t)));
    CUDA_CALL(cudaMalloc((void**)&dev_y_new, particle_count * sizeof(curandState_t)));
    CUDA_CALL(cudaMalloc((void**)&dev_t_new, particle_count * sizeof(curandState_t)));

    CUDA_CALL(cudaMalloc((void**)&dev_dx, bytes));
    CUDA_CALL(cudaMalloc((void**)&dev_dy, bytes));
    CUDA_CALL(cudaMalloc((void**)&dev_dt, bytes));

    CUDA_CALL(cudaMalloc((void**) &dev_states_x, particle_count * sizeof(curandState_t)));
    CUDA_CALL(cudaMalloc((void**) &dev_states_y, particle_count * sizeof(curandState_t)));
    CUDA_CALL(cudaMalloc((void**) &dev_states_theta, particle_count * sizeof(curandState_t)));

    CUDA_CALL(cudaMalloc((void**)&dev_deviation_x, bytes));
    CUDA_CALL(cudaMalloc((void**)&dev_deviation_y, bytes));
    CUDA_CALL(cudaMalloc((void**)&dev_deviation_theta, bytes));
    
    int grid_size = particle_count / MAX_THREAD + 1;
    normal_init_kernel<<<grid_size, MAX_THREAD>>>(dev_states_x, (float)time(NULL), particle_count);
    normal_init_kernel<<<grid_size, MAX_THREAD>>>(dev_states_y, (float)time(NULL), particle_count);
    normal_init_kernel<<<grid_size, MAX_THREAD>>>(dev_states_theta, (float)time(NULL), particle_count);

    set_fvalue_kernel<<<grid_size, MAX_THREAD>>>(dev_x, 0, particle_count);
    set_fvalue_kernel<<<grid_size, MAX_THREAD>>>(dev_y, 0, particle_count);
    set_fvalue_kernel<<<grid_size, MAX_THREAD>>>(dev_t, 0, particle_count);
    set_fvalue_kernel<<<grid_size, MAX_THREAD>>>(dev_w, 1.f, particle_count);
    
    this->x_x = this->x_y = this->x_r = 0;
    this->y_x = this->y_y = this->y_r = 0;
    this->r_x =  this->r_y =  this->r_r = 0;
    neff_thresh = 0*particle_count;
}

ParticleFilter::~ParticleFilter(){
    return;
}

void ParticleFilter::set_prediction_params(float x_x, float x_y, float x_r, float y_x, float y_y, float y_r, float r_x, float r_y, float r_r){
    this->x_x = x_x;
    this->x_y = x_y;
    this->x_r = x_r;

    this->y_x = y_x;
    this->y_y = y_y;
    this->y_r = y_r;

    this->r_x = r_x;
    this->r_y = r_y;
    this->r_r = r_r;
}

void ParticleFilter::set_correction_params(uint32_t corr_iter_thresh_num, int kernel_size){
    this->corr_iter_thresh_num = corr_iter_thresh_num;
    this->kernel_size = kernel_size;
}

void ParticleFilter::set_measurement_params(int _beam_count, float* _laser_angles, float _min_range, float _max_range){

    beam_count = _beam_count;
    min_range = _min_range;
    max_range = _max_range;

    CUDA_CALL(cudaMalloc((void**)&dev_scan_base_x, particle_count * beam_count*sizeof(float)));
    hst_scan_base_x = (float*)malloc(particle_count*beam_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_scan_base_y, particle_count * beam_count*sizeof(float)));
    hst_scan_base_y = (float*)malloc(particle_count*beam_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_scan_base_x_cell, particle_count*beam_count*sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&dev_scan_base_y_cell, particle_count*beam_count*sizeof(int)));

    CUDA_CALL(cudaMalloc((void**)&dev_scan_map_x, particle_count*beam_count*sizeof(float)));
    hst_scan_map_x = (float*)malloc(particle_count*beam_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_scan_map_y, particle_count*beam_count*sizeof(float)));
    hst_scan_map_y = (float*)malloc(particle_count*beam_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_corr_map_x, particle_count*beam_count*sizeof(float)));
    hst_corr_map_x = (float*)malloc(particle_count*beam_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_corr_map_y, particle_count*beam_count*sizeof(float)));
    hst_corr_map_y = (float*)malloc(particle_count*beam_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_corr_valids, particle_count*beam_count*sizeof(valid_t)));
    hst_corr_valids = (valid_t*)malloc(particle_count*beam_count*sizeof(valid_t));

    CUDA_CALL(cudaMalloc((void**)&dev_likelihoods, particle_count*beam_count*sizeof(float)));
    hst_likelihoods = (float*)malloc(particle_count*beam_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_like_valids, particle_count*beam_count*sizeof(valid_t)));
    hst_like_valids = (valid_t*)malloc(particle_count*beam_count*sizeof(valid_t));

    CUDA_CALL(cudaMalloc((void**)&dev_ranges, beam_count*sizeof(float)));

    CUDA_CALL(cudaMalloc((void**)&dev_angles, beam_count*sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_angles, _laser_angles, beam_count*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc((void**)&dev_ranges_valid, beam_count*sizeof(valid_t)));
    hst_ranges_valid = (valid_t*)malloc(beam_count*sizeof(valid_t));

    CUDA_CALL(cudaMalloc((void**)&dev_la, particle_count*sizeof(float)));
    hst_la = (float*)malloc(particle_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_c, particle_count*sizeof(float)));
    hst_c  = (float*)malloc(particle_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_s, particle_count*sizeof(float)));
    hst_s  = (float*)malloc(particle_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_tx, particle_count*sizeof(float))); 
    hst_tx = (float*)malloc(particle_count*sizeof(float));

    CUDA_CALL(cudaMalloc((void**)&dev_ty, particle_count*sizeof(float)));
    hst_ty = (float*)malloc(particle_count*sizeof(float));
}

void ParticleFilter::set_resample_params(float neff_thresh){
    this->neff_thresh = neff_thresh*particle_count;
}

void ParticleFilter::set_map_params(float global_map_width_m, float global_map_height_m, 
                                    float local_map_width_m, float local_map_height_m, 
                                    float true_positive, float true_negative, float w_thresh, 
                                    float m_per_cell){

    this->global_width_m = global_map_width_m;
    this->global_height_m = global_map_height_m;

    this->local_width_m = local_map_width_m;
    this->local_height_m = local_map_height_m;

    this->true_positive = true_positive;
    this->true_negative = true_negative;
    this->w_thresh = w_thresh;
    this->m_per_cell = m_per_cell;

    global_width_cell = ((int)(global_width_m/m_per_cell/2))*2;
    global_height_cell = ((int)(global_height_m/m_per_cell/2))*2;
    global_size_cell = global_width_cell * global_height_cell;
    CUDA_CALL(cudaMalloc((void**)&dev_global_map, particle_count * global_size_cell*sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&dev_global_map_new, particle_count * global_size_cell*sizeof(float)));
    hst_global_map = (float*)malloc(global_size_cell*sizeof(float));

    local_width_cell = ((int)(local_width_m/m_per_cell/2))*2;  
    local_height_cell = ((int)(local_height_m/m_per_cell/2))*2;
    local_size_cell = local_width_cell * local_height_cell;
    CUDA_CALL(cudaMalloc((void**)&dev_local_map, particle_count * local_size_cell*sizeof(float)));
    hst_local_map = (float*)malloc(local_size_cell*sizeof(float));

    ROS_INFO("Global map: %d %d %d", global_width_cell, global_height_cell, global_size_cell);
    ROS_INFO("Lobal map: %d %d %d", local_width_cell, local_height_cell, local_size_cell);
}

};