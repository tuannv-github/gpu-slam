#ifndef _PARTICLE_FILTER_H_
#define _PARTICLE_FILTER_H_
    
#include <curand_kernel.h>
#include <boost/thread.hpp>

#include <common.cuh>

namespace SLAM{

class ParticleFilter
{
private:

    /* Filter parameters */
    int particle_count;

    /* Prediction parameters */
    float x_x, x_y, x_r;
    float y_x, y_y, y_r;
    float r_x, r_y, r_r;

    /* Correction parameters */
    uint32_t corr_iter_thresh_num;
    int kernel_size;

    /* Measurement parameters */
    int beam_count;
    float min_range, max_range;

    /* Resample parameters*/
    float neff_thresh;

    /* Map parameters */
    float m_per_cell;
    float true_positive;
    float true_negative;
    float w_thresh;

    /* Filter internal variables */
    float *hst_x, *hst_y, *hst_t,  *hst_w;
    float *dev_x, *dev_y, *dev_t, *dev_w;
    float *dev_x_new, *dev_y_new, *dev_t_new;

    /* Origin of local map is in the center of the local map */
    float           *dev_local_map;
    float           *hst_local_map;
    boost::mutex    local_map_mutex;
    float           local_width_m;
    float           local_height_m;
    int             local_width_cell;
    int             local_height_cell;
    int             local_size_cell;
    float           local_origin_x_m;
    float           local_origin_y_m;

    /* Origin of global map is in the center of the global map */
    float           *dev_global_map;
    float           *hst_global_map;
    float           *dev_global_map_new;
    boost::mutex    global_map_mutex;
    float           global_width_m;
    float           global_height_m;
    int             global_width_cell;
    int             global_height_cell;
    int             global_size_cell;
    float           global_origin_x_m;
    float           global_origin_y_m;

    /* Prediction internal variables */
    float       *dev_dx, *dev_dy, *dev_dt;
    curandState *dev_states_x, *dev_states_y, *dev_states_theta;
    float       *dev_deviation_x, *dev_deviation_y, *dev_deviation_theta;

    /* Mesurement internal variables */
    float     *dev_ranges;
    float     *dev_angles;
    valid_t   *dev_ranges_valid;
    valid_t   *hst_ranges_valid;
    
    float *dev_scan_base_x, *dev_scan_base_y;
    float *hst_scan_base_x, *hst_scan_base_y;

    /* Correction internal variables */
    float *dev_scan_map_x, *dev_scan_map_y;
    float *hst_scan_map_x, *hst_scan_map_y;

    float *dev_corr_map_x, *dev_corr_map_y;
    float *hst_corr_map_x, *hst_corr_map_y;

    valid_t  *dev_corr_valids;
    valid_t  *hst_corr_valids;

    float *dev_likelihoods;
    float *hst_likelihoods;

    valid_t  *dev_like_valids;
    valid_t  *hst_like_valids;

    /* Transformation */
    float *hst_la, *hst_c, *hst_s, *hst_tx, *hst_ty;
    float *dev_la, *dev_c, *dev_s, *dev_tx, *dev_ty;

    /* Resampling internal variables */
    int     *dev_pick, *hst_pick;   // To pick state value in resampling 
    float   neff, max_w;
    int     max_w_idx;

    /* Local map update internal variables */
    int *dev_scan_base_x_cell, *dev_scan_base_y_cell; // dev_scan_base_x/m_per_cell, dev_scan_base_y/m_per_cell

    void    find_correspondace();
    int     calculate_transfomation();
    void    apply_transform();

public:
    ParticleFilter(int particle_count);
    ~ParticleFilter();

    /* Getter and setter */
    void get_x(float *&x, int &len);
    void get_y(float *&y, int &len);
    void get_t(float *&t, int &len);
    void get_w(float *&w, int &len);

    void set_x(float *x, int len);
    void set_y(float *y, int len);
    void set_t(float *t, int len);
    void set_w(float *w, int len);

    void get_local_map(uint32_t *_width_cell, uint32_t *_height_cell ,float *_m_per_cell, float *&_map);
    void get_local_map_idx(uint32_t *_width_cell, uint32_t *_height_cell ,float *_m_per_cell, float *&_map, int idx);

    void get_global_map(uint32_t *_width_cell, uint32_t *_height_cell ,float *_m_per_cell, float *&_map);
    void get_global_map_idx(uint32_t *_width_cell, uint32_t *_height_cell ,float *_m_per_cell, float *&_map, int idx);

    void get_particle_best(float *x, float *y, float *theta);
    void get_particle_idx(float *x, float *y, float *theta, int idx);

    void set_prediction_params(float x_x, float x_y, float x_r, float y_x, float y_y, float y_r, float r_x, float r_y, float r_r);
    void set_correction_params(uint32_t corr_iter_thresh_num, int kernel_size);
    void set_measurement_params(int beam_count, float* laser_angles, float min_range, float max_range);
    void set_resample_params(float neff_thresh);
    void set_map_params(float global_map_width_m, float global_map_height_m, 
                        float local_map_width_m, float local_map_height_m, 
                        float true_positive, float true_negative, float w_thresh, 
                        float m_per_cell);
    
    /* Filter initialization */
    bool set_initial_state(float x, float y, float t);      /* Initial state */

    /* Filter loop*/
    bool predict(float dx, float dy, float dtheta);         /* Prediction */
    bool set_measurement(const float *ranges);              /* Set measurement */
    bool correct();                                         /* Correction */
    bool score();                                           /* Score */
    bool resample();                                        /* Filter */
    bool resample_force();
    bool local_map_update();                                /* Update local occupancy using bresenham algorithm */
    bool global_map_update();                               /* Update map after filtering*/
    bool global_map_update_force();                         /* Update map after filtering*/
};

}

#endif
