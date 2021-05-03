#ifndef _SLAM_H_
#define _SLAM_H_

#include <stdint.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/msg/laser_scan.h>
#include <sensor_msgs/msg/imu.h>
#include <tf2_ros/message_filter.h>
#include <boost/thread.hpp>
#include <geometry_msgs/msg/pose_array.h>
#include <std_msgs/msg/float64.h>

#include "ParticleFilter.cuh"

namespace SLAM{

class Slam : public rclcpp::Node{

public:
    Slam();
    ~Slam();

    bool start_live_slam();
    bool stop_live_slam();

private:
    ros::NodeHandle nh;         // node handler
    ros::NodeHandle pnh;        // private node handler

    std::string map_frame;      // map frame
    std::string odom_frame;     // odometry frame
    std::string urbase_frame;   // un-rotated base frame
    std::string base_frame;     // robot frame
    std::string laser_frame;    // laser frame
    std::string laser_topic;    // laser topic
    std::string particle_topic; // particle topic

    /* Prediction parameters */
    float x_x, x_y, x_r;
    float y_x, y_y, y_r;
    float r_x, r_y, r_r;
    float old_x, old_y, old_t;

    /* Set measurement parameters */
    int     laser_count;
    float   min_range, max_range;

    /* Correction parameters */
    int corr_iter_thresh_num;
    float kernel_size_m;

    /* Resampling parameters */
    float neff_thresh;
    float dx_resample_thresh;
    float dy_resample_thresh;
    float dt_resample_thresh;

    /* Map parameters */
    float m_per_cell;
    float initial_local_map_width;
    float initial_local_map_height;
    float initial_global_map_width;
    float initial_global_map_height;
    float true_positive;
    float true_negative;
    float w_thresh;
    int initialization_scan;
    float dx_map_update_thresh;
    float dy_map_update_thresh;
    float dt_map_update_thresh;

    tf::TransformListener       tf_listener;
    tf::TransformBroadcaster*   tf_broadcaster;

    tf::Transform   map_to_odom;            boost::mutex    map_to_odom_mutex;
    tf::Transform   odom_to_urbase;
    tf::Transform   odom_to_base;
    tf::Transform   base_to_laser;

    ros::Publisher global_map_plr;  ros::Publisher global_map_metadata_plr; nav_msgs::GetMap::Response global_map_msg;  boost::mutex global_map_mutex;   
    ros::Publisher local_map_plr;   ros::Publisher local_map_metadata_plr;  nav_msgs::GetMap::Response local_map_msg;   boost::mutex local_map_mutex;   
    ros::Publisher particle_plr;    geometry_msgs::PoseArray partiles_msg;
    ros::Publisher fps_plr;

    boost::thread *map_to_odom_plr_thread;      void map_to_odom_pl_thread();       float map_to_odom_pl_period;
    boost::thread *map_to_urbase_plr_thread;    void map_to_urbase_pl_thread();     float map_to_urbase_pl_period;
    boost::thread *global_map_plr_thread;       void global_map_pl_thread();        float global_map_pl_period;
    boost::thread *local_map_plr_thread;        void local_map_pl_thread();         float local_map_pl_period;
    boost::thread *particle_plr_thread;         void particle_pl_thread();          float particle_pl_period;

    message_filters::Subscriber<sensor_msgs::LaserScan>* laser_scan_subscriber;
    tf::MessageFilter<sensor_msgs::LaserScan>* laser_scan_filter;
    void on_range_msg_callback(const sensor_msgs::LaserScan::ConstPtr& scan_msg);

    /* Filter variables */
    int particle_count;
    ParticleFilter *particle_filter;
    bool is_first_scan, got_new_global_map, got_new_local_map;

    ros::WallTime start_, end_;

    /* First scan initial function */
    bool init_initial_state(const sensor_msgs::LaserScan::ConstPtr& scan_msg);
    bool init_correction_params();
    bool init_mesurement_params(const sensor_msgs::LaserScan::ConstPtr& scan_msg);
    bool init_resample_params();
    bool init_map_params();

    bool get_movement(float &dx, float &dy, float &dt, const sensor_msgs::LaserScan::ConstPtr &scan_msg);
};

};

#endif