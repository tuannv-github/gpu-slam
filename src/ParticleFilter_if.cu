#include <time.h>
#include <stdlib.h>

#include <ParticleFilter.cuh>
#include <ParticleFilter_kernel.cuh>
#include <CudaUtils.cuh>

namespace SLAM{

    void ParticleFilter::get_x(float *&x, int &len){
        CUDA_CALL(cudaMemcpy(hst_x, dev_x, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        x = hst_x;
        len = particle_count;
    }
    
    void ParticleFilter::get_y(float *&y, int &len){
        CUDA_CALL(cudaMemcpy(hst_y, dev_y, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        y = hst_y;
        len = particle_count;
    }
    
    void ParticleFilter::get_t(float *&t, int &len){
        CUDA_CALL(cudaMemcpy(hst_t, dev_t, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        t = hst_t;
        len = particle_count;
    }

    void ParticleFilter::get_w(float *&w, int &len){
        CUDA_CALL(cudaMemcpy(hst_w, dev_w, particle_count * sizeof(float), cudaMemcpyDeviceToHost));
        w = hst_w;
        len = particle_count;
    }

    void ParticleFilter::set_x(float *x, int len){
        CUDA_CALL(cudaMemcpy(dev_x, x, len*sizeof(float), cudaMemcpyHostToDevice));
    }
    
    void ParticleFilter::set_y(float *y, int len){
        CUDA_CALL(cudaMemcpy(dev_y, y, len*sizeof(float), cudaMemcpyHostToDevice));
    }
    
    void ParticleFilter::set_t(float *t, int len){
        CUDA_CALL(cudaMemcpy(dev_t, t, len*sizeof(float), cudaMemcpyHostToDevice));
    }

    void ParticleFilter::set_w(float *w, int len){
        CUDA_CALL(cudaMemcpy(dev_w, w, len*sizeof(float), cudaMemcpyHostToDevice));
    }

    void ParticleFilter::get_local_map(uint32_t *_width_cell, uint32_t *_height_cell ,float *_m_per_cell, float *&_map){

        // CUDA_CALL(cudaMemcpy(hst_w, dev_w, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        int idx = 0;
        float max = hst_w[0];
        for(int i=1; i < particle_count; i++){
            if(hst_w[i] >= max){
                max = hst_w[i];
                idx = i;
            }
        }
    
        local_map_mutex.lock();
        CUDA_CALL(cudaMemcpy(hst_local_map, dev_local_map + idx*local_size_cell, local_size_cell*sizeof(float), cudaMemcpyDeviceToHost));
        local_map_mutex.unlock();
    
        _map = hst_local_map;
        *_width_cell = local_width_cell;
        *_height_cell = local_height_cell;
        *_m_per_cell = m_per_cell;
    }

    void ParticleFilter::get_local_map_idx(uint32_t *_width_cell, uint32_t *_height_cell ,float *_m_per_cell, float *&_map, int idx){
        local_map_mutex.lock();
        CUDA_CALL(cudaMemcpy(hst_local_map, dev_local_map + idx*local_size_cell, local_size_cell*sizeof(float), cudaMemcpyDeviceToHost));
        local_map_mutex.unlock();
    
        _map = hst_local_map;
        *_width_cell = local_width_cell;
        *_height_cell = local_height_cell;
        *_m_per_cell = m_per_cell;
    }

    void ParticleFilter::get_global_map(uint32_t *_width_cell, uint32_t *_height_cell ,float *_m_per_cell, float *&_map){
        
        // CUDA_CALL(cudaMemcpy(hst_w, dev_w, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        int idx = 0;
        float max = hst_w[0];
        for(int i=1; i < particle_count; i++){
            if(hst_w[i] >= max){
                max = hst_w[i];
                idx = i;
            }
        }
    
        global_map_mutex.lock();
        CUDA_CALL(cudaMemcpy(hst_global_map, dev_global_map + idx*global_size_cell, global_size_cell*sizeof(float), cudaMemcpyDeviceToHost));
        global_map_mutex.unlock();
    
        _map = hst_global_map;
        *_width_cell = global_width_cell;
        *_height_cell = global_height_cell;
        *_m_per_cell = m_per_cell;
    }

    void ParticleFilter::get_particle_best(float *x, float *y, float *t){
    
        int idx = 0;
        float max = hst_w[0];
        for(int i=1; i < particle_count; i++){
            if(hst_w[i] >= max){
                max = hst_w[i];
                idx = i;
            }
        }
    
        CUDA_CALL(cudaMemcpy(hst_x, dev_x, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_y, dev_y, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_t, dev_t, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
    
        *x = hst_x[idx];
        *y = hst_y[idx];
        *t = hst_t[idx];
    }

    void ParticleFilter::get_particle_idx(float *x, float *y, float *t, int idx){
    
        CUDA_CALL(cudaMemcpy(hst_x, dev_x, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_y, dev_y, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_t, dev_t, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
    
        *x = hst_x[idx];
        *y = hst_y[idx];
        *t = hst_t[idx];
    }
    
    void ParticleFilter::get_global_map_idx(uint32_t *_width_cell, uint32_t *_height_cell ,float *_m_per_cell, float *&_map, int idx){
        
        global_map_mutex.lock();
        CUDA_CALL(cudaMemcpy(hst_global_map, dev_global_map + idx*global_size_cell, global_size_cell*sizeof(float), cudaMemcpyDeviceToHost));
        global_map_mutex.unlock();
    
        _map = hst_global_map;
        *_width_cell = global_width_cell;
        *_height_cell = global_height_cell;
        *_m_per_cell = m_per_cell;
    }

}