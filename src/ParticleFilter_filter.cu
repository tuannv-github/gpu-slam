#include <time.h>
#include <stdlib.h>

#include <ParticleFilter.cuh>
#include <ParticleFilter_kernel.cuh>
#include <CudaUtils.cuh>

namespace SLAM{

    bool ParticleFilter::set_initial_state(float x, float y, float t){
        int grid_size = particle_count;
        float w = 1.0;
        set_fvalue_kernel<<<grid_size, MAX_THREAD>>>(dev_x, x, particle_count);
        set_fvalue_kernel<<<grid_size, MAX_THREAD>>>(dev_y, y, particle_count);
        set_fvalue_kernel<<<grid_size, MAX_THREAD>>>(dev_t, t, particle_count); 
        set_fvalue_kernel<<<grid_size, MAX_THREAD>>>(dev_w, w, particle_count); 

        #ifdef DEBUG_INITIALIZATION
            CUDA_CALL(cudaMemcpy(hst_x, dev_x, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(hst_y, dev_y, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(hst_t, dev_t, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(hst_w, dev_w, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
            printf("----------------------------->>> Initialization\n");
            printf("(x, y, t, w)\n");
            for(int i=0; i<particle_count; i++){
                printf("(%f, %f, %f, %f) \n", hst_x[i], hst_y[i], hst_t[i], hst_w[i]);
            }
        #endif

        return true;
    }

    bool ParticleFilter::predict(float dx, float dy, float dt){

        int grid_size = particle_count / MAX_THREAD + 1;
    
        prediction_kernel<<<grid_size, MAX_THREAD>>>(dev_x, dev_y, dev_t, 
                                                    dx, dy, dt,
                                                    x_x*dx + x_y*dy + x_r*dt, y_x*dx + y_y*dy + y_r*dt,  r_x*dx + r_y*dy + r_r*dt,
                                                    particle_count);
        
        #ifdef DEBUG_PREDICTION
            CUDA_CALL(cudaMemcpy(hst_x, dev_x, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(hst_y, dev_y, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(hst_t, dev_t, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
            printf("----------------------------->>> Prediction\n");
            int i;
            for(i=0; i<particle_count; i++){
                printf("%f\t", hst_x[i]);
                if(i%10==9) printf("\n");
            }
            if(i%10) printf("\n");
            for(i=0; i<particle_count; i++){
                printf("%f\t", hst_y[i]);
                if(i%10==9) printf("\n");
            }
            if(i%10) printf("\n");
            for(i=0; i<particle_count; i++){
                printf("%f\t", hst_t[i]);
                if(i%10==9) printf("\n");
            }
            if(i%10) printf("\n");
        #endif

        return true;
    }

    bool ParticleFilter::set_measurement(const float *ranges){
        // Update mesurement
        CUDA_CALL(cudaMemcpy(dev_ranges, ranges, beam_count*sizeof(float), cudaMemcpyHostToDevice));

        // Check measuement valid
        int grid_size = beam_count/MAX_THREAD + 1;
        ranges_threshold_kernel<<<grid_size, MAX_THREAD>>>(dev_ranges, dev_ranges_valid, min_range, max_range, beam_count);

        #ifdef DEBUG_SET_MEASUREMENT
        CUDA_CALL(cudaMemcpy(hst_ranges_valid, dev_ranges_valid, beam_count*sizeof(valid_t), cudaMemcpyDeviceToHost));
        printf("----------------------------->>> Mesurement\n");
        int i;
        for(i=0; i<beam_count; i++){
            printf("%d\t", hst_ranges_valid[i]);
            if(i%10==9) printf("\n");
        }
        #endif

        return true;
    }

    void ParticleFilter::find_correspondace(){

        int block_size, grid_size;

        #ifdef DEBUG_CORRECTION
        int i;
        printf("-------------------------------------------------------------------------------------------------------------------------------------------------\n");
        printf("----------------------------->>>  State\n");
        CUDA_CALL(cudaMemcpy(hst_x, dev_x, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_y, dev_y, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_t, dev_t, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        for(i=0; i<particle_count; i++){
            printf("(%f, %f, %f)\t", hst_x[i], hst_y[i], hst_t[i]);
            if(i%3==2) printf("\n");
        }
        if(i%3) printf("\n");
        printf("----------------------------->>>  End of state\n");
        #endif

        block_size = MAX_THREAD;
        grid_size  = beam_count / block_size + 1;
        grid_size = particle_count * beam_count / block_size + 1;
        scan_base_polar_to_cart_kernel<<<grid_size, block_size>>>(  dev_scan_base_x, dev_scan_base_y,
                                                                    dev_t, dev_ranges, dev_angles, 
                                                                    dev_ranges_valid,  beam_count,
                                                                    particle_count*beam_count);
        
        #ifdef DEBUG_CORRECTION
        printf("----------------------------->>>  Scan base polar to cart\n");
        CUDA_CALL(cudaMemcpy(hst_scan_base_x, dev_scan_base_x, beam_count*particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_scan_base_y, dev_scan_base_y, beam_count*particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        for(int particle_idx=0; particle_idx<particle_count; particle_idx++){
            printf("Partile %d:\n", particle_idx);
            for(i=0; i<beam_count; i++){
                printf("(%f, %f)\t", hst_scan_base_x[i+particle_idx*beam_count], hst_scan_base_y[i+particle_idx*beam_count]);
                if(i%5==4) printf("\n");
            }
            if(i%5) printf("\n");
        }
        printf("----------------------------->>>  End of scan base polar to cart\n");
        #endif

        scan_base_to_map_kernel<<<grid_size, block_size>>>( dev_scan_map_x, dev_scan_map_y,
                                                            dev_scan_base_x, dev_scan_base_y, dev_ranges_valid,
                                                            dev_x, dev_y, beam_count,
                                                            particle_count*beam_count);
        
        CUDA_CALL(cudaMemcpy(hst_scan_map_x, dev_scan_map_x, particle_count*beam_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_scan_map_y, dev_scan_map_y, particle_count*beam_count*sizeof(float), cudaMemcpyDeviceToHost));

        #ifdef DEBUG_CORRECTION
        printf("----------------------------->>>  Scan base to map\n");

        for(int particle_idx=0; particle_idx<particle_count; particle_idx++){
            printf("Partile %d:\n", particle_idx);
            for(i=0; i<beam_count; i++){
                printf("(%f, %f)\t", hst_scan_map_x[i+particle_idx*beam_count], hst_scan_map_y[i+particle_idx*beam_count]);
                if(i%5==4) printf("\n");
            }
            if(i%5) printf("\n");
        }
        #endif

        global_map_mutex.lock();
        find_correspondence_kernel<<<grid_size, block_size>>>(  dev_corr_map_x, dev_corr_map_y, dev_corr_valids,
                                                                dev_scan_map_x, dev_scan_map_y, dev_ranges_valid, m_per_cell,
                                                                dev_global_map, global_width_cell, global_height_cell, global_size_cell,
                                                                beam_count, kernel_size,
                                                                beam_count*particle_count);
        global_map_mutex.unlock();

        CUDA_CALL(cudaMemcpy(hst_corr_map_x, dev_corr_map_x, beam_count*particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_corr_map_y, dev_corr_map_y, beam_count*particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_corr_valids, dev_corr_valids, beam_count*particle_count*sizeof(valid_t), cudaMemcpyDeviceToHost));
        
        // // One - one corresspondance
        // for(int k=0; k<particle_count; k++){
        //     float *x = hst_corr_map_x + k*beam_count;
        //     float *y = hst_corr_map_y + k*beam_count;
        //     valid_t *v = hst_corr_valids + k*beam_count;
        //     for(int i=0; i<beam_count; i++){
        //         for(int j=i+1; j<beam_count; j++){
        //             if(x[i]==x[j] && y[i]==y[j]){
        //                v[j] = VVLD;
        //             }
        //         }
        //     }
        // }

        #ifdef DEBUG_CORRECTION
        printf("----------------------------->>>  Correspondance\n");
        for(int particle_idx=0; particle_idx<particle_count; particle_idx++){
            printf("Partile %d:\n", particle_idx);
            for(i=0; i<beam_count; i++){
                printf("%f, %f\t%f, %f\t%d\n", 
                hst_scan_map_x[i+particle_idx*beam_count], hst_scan_map_y[i+particle_idx*beam_count], 
                hst_corr_map_x[i+particle_idx*beam_count], hst_corr_map_y[i+particle_idx*beam_count], 
                hst_corr_valids[i+particle_idx*beam_count]) ;
            }
        }
        printf("----------------------------->>>  End of correspondance\n");
        #endif
    }

    static void calc_transform( float &la, float &c, float &s, float &tx, float &ty, 
                                float *l_x, float *l_y, float *r_x, float *r_y, valid_t *valids, uint32_t len){

        /* Compute the mean */
        float l_x_avg=0, l_y_avg=0, r_x_avg=0, r_y_avg=0;
        int count=0;
        for(int i=0; i<len; i++){
            if(valids[i] != VVLD) continue;
            l_x_avg += l_x[i];
            l_y_avg += l_y[i];
            r_x_avg += r_x[i];
            r_y_avg += r_y[i];
            count++;
        }
        // printf("count %d\n", count);
        count = count == 0 ? 1 : count;
        l_x_avg/=(float)count; l_y_avg/=(float)count; r_x_avg/=(float)count; r_y_avg/=(float)count; 
        // printf("%f %f %f %f\n", l_x_avg, l_y_avg, r_x_avg, r_y_avg);    

        /* Move to reduced coordinate */
        for(int i=0; i<len; i++){
            if(valids[i] != VVLD) continue;
            l_x[i] -= l_x_avg;
            l_y[i] -= l_y_avg;
            r_x[i] -= r_x_avg;
            r_y[i] -= r_y_avg;
        } 

        /* Calculate temporary variables */
        float cs=0, ss=0, rr=0, ll=0;
        for(int i=0; i<len; i++){
            if(valids[i] != VVLD) continue;
            float lx=l_x[i], ly=l_y[i];
            float rx=r_x[i], ry=r_y[i];
            cs +=   rx*lx + ry*ly;
            ss +=  -rx*ly + ry*lx;
            rr +=   rx*rx + ry*ry;
            ll +=   lx*lx + ly*ly;
        }
        // printf("%f %f %f %f\n", cs, ss, rr, ll);    
            
        /* Caculate transform */
        la = sqrt(rr/ll);
        float msos = sqrt(cs*cs + ss*ss);
        c = cs/msos;
        s = ss/msos;
        // printf("%f %f %f\n", la, c, s);  

        tx = r_x_avg - la*(c*l_x_avg - s*l_y_avg);
        ty = r_y_avg - la*(s*l_x_avg + c*l_y_avg);

        // printf("%f %f %f %f %f\n", la, c, s, tx, ty);
        // printf("-----------------------------\n"); 
    }

    int ParticleFilter::calculate_transfomation(){

        int cnt = 0;
        for(int particle_idx=0; particle_idx<particle_count; particle_idx++){
            float   *l_x = hst_scan_map_x + particle_idx*beam_count;
            float   *l_y = hst_scan_map_y + particle_idx*beam_count;
            float   *r_x = hst_corr_map_x + particle_idx*beam_count;
            float   *r_y = hst_corr_map_y + particle_idx*beam_count;
            valid_t *valids = hst_corr_valids + particle_idx*beam_count;

            calc_transform( hst_la[particle_idx], hst_c[particle_idx], hst_s[particle_idx], hst_tx[particle_idx], hst_ty[particle_idx],
                            l_x, l_y, r_x, r_y, valids, beam_count);
        }
        return cnt;
    }

    void ParticleFilter::apply_transform(){

        CUDA_CALL(cudaMemcpy(dev_la, hst_la, particle_count*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dev_c, hst_c, particle_count*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dev_s, hst_s, particle_count*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dev_tx, hst_tx, particle_count*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dev_ty, hst_ty, particle_count*sizeof(float), cudaMemcpyHostToDevice));

        dim3 grid_size(particle_count / MAX_BLOCK_SIZE + 1);
        dim3 block_size(MAX_BLOCK_SIZE);
        apply_transform_kernel<<<grid_size, block_size>>>(dev_x, dev_y, dev_t,
                                                            dev_la, dev_c, dev_s, dev_tx, dev_ty,
                                                            particle_count);
    }
    
    bool ParticleFilter::correct(){
        
        int cnt = 0;
        while( cnt < corr_iter_thresh_num){
            find_correspondace();
            calculate_transfomation();
            apply_transform();
            cnt++; 
        }
        
        #ifdef DEBUG_CORRECTION
        int i;
        printf("----------------------------->>>  Last transfomation\n");
        printf("This is the %d transformation\n", cnt);
        for(i=0; i <particle_count; i++){
            printf("(%f %f %f %f %f)\t", hst_la[i], hst_c[i], hst_s[i], hst_tx[i], hst_ty[i]);
            if(i%2==1) printf("\n");
        }
        if(i%2) printf("\n");
        printf("----------------------------->>>  End of last transfomation\n");
        printf("----------------------------->>>  State after correction\n");
        CUDA_CALL(cudaMemcpy(hst_x, dev_x, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_y, dev_y, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_t, dev_t, particle_count*sizeof(float), cudaMemcpyDeviceToHost));
        for(i=0; i<particle_count; i++){
            printf("(%f, %f, %f)\t", hst_x[i], hst_y[i], hst_t[i]);
            if(i%3==2) printf("\n");
        }
        if(i%3) printf("\n");
        printf("----------------------------->>>  End of state after correction\n");
        #endif

        return true;
    }

    bool ParticleFilter::score(){
        
        #ifdef DEBUG_SCORE
            printf("----------------------------->>>  Score\n");
        #endif

        int block_size, grid_size;

        block_size = MAX_THREAD;
        grid_size = particle_count * beam_count / block_size + 1;
    
        scan_base_polar_to_cart_kernel<<<grid_size, block_size>>>(  dev_scan_base_x, dev_scan_base_y,
                                                                    dev_t, dev_ranges, dev_angles, 
                                                                    dev_ranges_valid,  beam_count,
                                                                    particle_count*beam_count);

        scan_base_to_map_kernel<<<grid_size, block_size>>>( dev_scan_map_x, dev_scan_map_y,
                                                            dev_scan_base_x, dev_scan_base_y, dev_ranges_valid,
                                                            dev_x, dev_y, beam_count,
                                                            particle_count*beam_count);
        
        global_map_mutex.lock();
        likelihood_kernel<<<grid_size, MAX_THREAD>>>(   dev_likelihoods, dev_like_valids,
                                                        dev_global_map, m_per_cell, dev_ranges_valid,
                                                        global_width_cell, global_height_cell, global_size_cell,
                                                        dev_x, dev_y, particle_count, beam_count,
                                                        dev_scan_map_x, dev_scan_map_y, particle_count * beam_count);
        global_map_mutex.unlock();

        cudaDeviceSynchronize();

        CUDA_CALL(cudaMemcpy(hst_likelihoods, dev_likelihoods, particle_count * beam_count *sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(hst_like_valids, dev_like_valids, particle_count * beam_count *sizeof(valid_t), cudaMemcpyDeviceToHost));
        
        int i;
        float min = std::numeric_limits<float>::max(); 
        for(i=0; i<particle_count; i++){   
            hst_w[i] = 0;
            for(int j=0; j<beam_count; j++){
                hst_w[i] += hst_likelihoods[i*beam_count + j];
            }
            hst_w[i]/=beam_count;
            min = min > hst_w[i] ? hst_w[i] : min;

            #ifdef DEBUG_SCORE
                printf("%f\t", hst_w[i]);
                if(i%10==9) printf("\n");
            #endif
        }

        CUDA_CALL(cudaMemcpy(dev_w, hst_w, particle_count*sizeof(float), cudaMemcpyHostToDevice));

        #ifdef DEBUG_SCORE
            if(i%10) printf("\n");
            printf("min: %f\n",min);
        #endif
        
        float sum = 0;
        for(int i=0; i<particle_count; i++){
            hst_w[i] -= min;
            sum += hst_w[i];

            #ifdef DEBUG_SCORE
                printf("%f\t", hst_w[i]);
                if(i%10==9) printf("\n");
            #endif
        }

        #ifdef DEBUG_SCORE
            if(i%10) printf("\n");
            printf("sum: %f\n",sum);
        #endif
        
        if(sum > 0.01f){
            max_w = std::numeric_limits<float>::min();
            max_w_idx = 0;
            for(i=0; i<particle_count; i++){
                hst_w[i] /= sum;
                if(hst_w[i] > max_w){
                    max_w = hst_w[i];
                    max_w_idx = i;
                }
                #ifdef DEBUG_SCORE
                    printf("%f\t", hst_w[i]);
                    if(i%10==9) printf("\n");
                #endif
            }
        }
        else{
            for(i=0; i<particle_count; i++){
                hst_w[i] = 1.0f/particle_count;
                #ifdef DEBUG_SCORE
                    printf("%f\t", hst_w[i]);
                    if(i%10==9) printf("\n");
                #endif
            }
            max_w = 1;
            max_w_idx = 0;
        }
        
        #ifdef DEBUG_SCORE
            if(i%10) printf("\n");
        #endif

        float sos = 0;
        sos += hst_w[0]*hst_w[0];
        for(i=1; i<particle_count; i++){
            sos += hst_w[i]*hst_w[i];
        }
        neff = 1.0/sos;

        #ifdef DEBUG_SCORE
            printf("neff %f vs %f \n", neff, neff_thresh);
        #endif

        return true;
    }

    float random_u(float min, float max) {
        srand( time(NULL) );
        float r = ((float)rand()) / (float) RAND_MAX;
        float diff = max - min;
        r = r * diff;
        return min + r;
    }

    bool ParticleFilter::resample_force(){
        #ifdef DEBUG_RESAMPLE
        printf("Resample force\n");
        #endif

        float  c[particle_count];
        c[0] = hst_w[0];
        for(int i=1; i<particle_count; i++){
            c[i] = c[i-1] + hst_w[i];
        }

        float u = random_u(0,1.f/particle_count);
        for(int i=0, j=0; i<particle_count; i++){
            while(u > c[j]) j++;
            hst_pick[i]=j;
            u += 1.0/particle_count;
        }
    
        CUDA_CALL(cudaMemcpy(dev_pick, hst_pick, particle_count, cudaMemcpyHostToDevice));
    
        dim3 grid_size(particle_count/MAX_THREAD + 1);
        dim3 block_size(MAX_THREAD);
    
        copy_state_kernel<<<grid_size, block_size>>>(dev_x_new, dev_x, dev_pick, particle_count);
        copy_state_kernel<<<grid_size, block_size>>>(dev_y_new, dev_y, dev_pick, particle_count);
        copy_state_kernel<<<grid_size, block_size>>>(dev_t_new, dev_t, dev_pick, particle_count);
    
        for(int i=0; i<particle_count; i++){
            grid_size = global_size_cell/MAX_THREAD + 1;
            block_size = (MAX_THREAD);
            copy_map_kernel<<<grid_size, block_size>>>(dev_global_map_new + i*global_size_cell,
                                                        dev_global_map + hst_pick[i]*global_size_cell, 
                                                        global_size_cell);
        }
    
        float *temp;
    
        temp = dev_x_new;
        dev_x_new = dev_x;
        dev_x = temp;
    
        temp = dev_y_new;
        dev_y_new = dev_y;
        dev_y = temp;
    
        temp = dev_t_new;
        dev_t_new = dev_t;
        dev_t = temp;
    
        temp = dev_global_map_new;
        dev_global_map_new = dev_global_map;
        dev_global_map = temp;
    
        return true;
    }
    
    bool ParticleFilter::resample(){

        if(neff > neff_thresh) return false;
        #ifdef DEBUG_RESAMPLE
        printf("----------------------------->>> Resampled\n");
        printf("neff %f\tneff_thresh %f\n", neff, neff_thresh);
        #endif
        return resample_force();
    }

    bool ParticleFilter::local_map_update(){
        int grid_size;
        grid_size = local_size_cell*particle_count / MAX_THREAD + 1;
        set_fvalue_kernel<<<grid_size, MAX_THREAD>>>(dev_local_map, 0, local_size_cell*particle_count);
    
        grid_size = beam_count*particle_count / MAX_THREAD + 1;
        set_ivalue_kernel<<<grid_size, MAX_THREAD>>>(dev_scan_base_x_cell, 0, beam_count*particle_count);
        set_ivalue_kernel<<<grid_size, MAX_THREAD>>>(dev_scan_base_y_cell, 0, beam_count*particle_count);
    
        grid_size = beam_count*particle_count / MAX_THREAD + 1;
        base_polar_to_cart_local_kernel<<<grid_size, MAX_THREAD>>>(dev_ranges, dev_angles, dev_ranges_valid, beam_count,
                                                                    dev_t, particle_count,
                                                                    dev_scan_base_x_cell, dev_scan_base_y_cell,
                                                                    m_per_cell, particle_count*beam_count);

        grid_size = particle_count*beam_count/MAX_THREAD + 1;
        local_map_mutex.lock();
        bresenham_occupancy_update_kernel<<<grid_size, MAX_THREAD>>>(dev_local_map, local_width_cell, local_height_cell,
                                                                    0,0, beam_count,
                                                                    dev_scan_base_x_cell, dev_scan_base_y_cell, dev_ranges_valid,
                                                                    true_positive, true_negative, particle_count*beam_count);
        local_map_mutex.unlock();

        return true;
    }

    bool ParticleFilter::global_map_update(){

        int width = particle_count*local_width_cell/MAX_BLOCK_SIZE + 1;
        int height = local_height_cell/MAX_BLOCK_SIZE + 1;
        dim3 grid(width, height);
        dim3 block(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
    
        add_local_map_to_global_map_kernel<<<grid,block>>>(dev_global_map, global_width_cell, global_height_cell,
                                                            dev_local_map, local_width_cell, local_height_cell,
                                                            dev_x, dev_y,
                                                            m_per_cell, local_size_cell, global_size_cell,
                                                            dev_w, w_thresh,
                                                            particle_count, particle_count*local_width_cell, local_height_cell);
        
        #ifdef DEBUG_MAP_UPDATE
        printf("----------------------------->>> Map updated\n");
        #endif 

        return true;
    }

    bool ParticleFilter::global_map_update_force(){

        int width = particle_count*local_width_cell/MAX_BLOCK_SIZE + 1;
        int height = local_height_cell/MAX_BLOCK_SIZE + 1;
        dim3 grid(width, height);
        dim3 block(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
    
        add_local_map_to_global_map_kernel<<<grid,block>>>(dev_global_map, global_width_cell, global_height_cell,
                                                            dev_local_map, local_width_cell, local_height_cell,
                                                            dev_x, dev_y,
                                                            m_per_cell, local_size_cell, global_size_cell,
                                                            dev_w, 0,
                                                            particle_count, particle_count*local_width_cell, local_height_cell);
        
        #ifdef DEBUG_MAP_UPDATE
        printf("----------------------------->>> Map updated\n");
        #endif 

        return true;
    }

};