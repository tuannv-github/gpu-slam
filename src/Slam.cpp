#include <Slam.h>
#include <nav_msgs/GetMap.h>
#include <math.h>
#include <chrono>
namespace SLAM{

Slam::Slam():
  pnh("~")
{
    /* Initialize parameters */
    if(!pnh.getParam("map_frame", map_frame))
        map_frame = "map";
    if(!pnh.getParam("odom_frame", odom_frame))
        odom_frame = "odom";
    if(!pnh.getParam("urbase_frame", urbase_frame))
        urbase_frame = "urbase_link";
    if(!pnh.getParam("base_frame", base_frame))
        base_frame = "base_link";
    if(!pnh.getParam("laser_frame", laser_frame))
        laser_frame = "laser_link";
    if(!pnh.getParam("laser_topic", laser_topic))
        laser_topic = "scan";
    if(!pnh.getParam("particle_topic", particle_topic))
        particle_topic = "particles";

    if(!pnh.getParam("map_to_odom_publish_period", map_to_odom_pl_period))
        map_to_odom_pl_period = 0.1;
    if(!pnh.getParam("map_to_urbase_pl_period", map_to_urbase_pl_period))
        map_to_urbase_pl_period = 0.1;
    if(!pnh.getParam("global_map_publish_period", global_map_pl_period))
        global_map_pl_period = 1;
    if(!pnh.getParam("local_map_publish_period", local_map_pl_period))
        local_map_pl_period = 0.5;
    if(!pnh.getParam("particle_pl_period", particle_pl_period))
        particle_pl_period = 0.5;

    if(!pnh.getParam("particle_count", particle_count))
        particle_count = 10;
    if(!pnh.getParam("initialization_scan", initialization_scan))
        initialization_scan = 30;

    if(!pnh.getParam("x_x", x_x))
        x_x = 0.01;
    if(!pnh.getParam("x_y", x_y))
        x_y = 0.01;
    if(!pnh.getParam("x_r", x_r))
        x_r = 0.01;

    if(!pnh.getParam("y_x", y_x))
        y_x = 0.01;
    if(!pnh.getParam("y_y", y_y))
        y_y = 0.01;
    if(!pnh.getParam("y_r", y_r))
        y_r = 0.01;

    if(!pnh.getParam("r_x", r_x))
        r_x = 0.01;
    if(!pnh.getParam("r_y", r_y))
        r_y = 0.01;
    if(!pnh.getParam("r_r", r_r))
        r_r = 0.01;

    if(!pnh.getParam("corr_iter_thresh_num", corr_iter_thresh_num))
        corr_iter_thresh_num = 5;
    if(!pnh.getParam("kernel_size_m", kernel_size_m))
        kernel_size_m = 0.1;

    if(!pnh.getParam("neff_thresh", neff_thresh))
        neff_thresh = 0.5;
    if(!pnh.getParam("dx_resample_thresh", dx_resample_thresh))
        dx_resample_thresh = 1;
    if(!pnh.getParam("dy_resample_thresh", dy_resample_thresh))
        dy_resample_thresh = 1;
    if(!pnh.getParam("dt_resample_thresh", dt_resample_thresh))
        dt_resample_thresh = 1;

    if(!pnh.getParam("min_range", min_range))
        min_range = 0.1;
    if(!pnh.getParam("max_range", max_range))
        max_range = 3.5;

    if(!pnh.getParam("initial_global_map_width", initial_global_map_width))
        initial_global_map_width = 50;
    if(!pnh.getParam("initial_global_map_height", initial_global_map_height))
        initial_global_map_height = 50;
    if(!pnh.getParam("initial_local_map_width", initial_local_map_width))
        initial_local_map_width = 10;
    if(!pnh.getParam("initial_local_map_height", initial_local_map_height))
        initial_local_map_height = 10;
    if(!pnh.getParam("m_per_cell", m_per_cell))
        m_per_cell = 0.05;
    if(!pnh.getParam("true_positive", true_positive))
        true_positive = 0.9;
    if(!pnh.getParam("true_negative", true_negative))
        true_negative = 0.5;
    if(!pnh.getParam("w_thresh", w_thresh))
        w_thresh = 0.2;
    if(!pnh.getParam("dx_map_update_thresh", dx_map_update_thresh))
        dx_map_update_thresh = 0.02;
    if(!pnh.getParam("dy_map_update_thresh", dy_map_update_thresh))
        dy_map_update_thresh = 0.02;
    if(!pnh.getParam("dt_map_update_thresh", dt_map_update_thresh))
        dt_map_update_thresh = 0.02;

    ROS_DEBUG("Slam initialized");
}

Slam::~Slam(){

}

bool Slam::start_live_slam(){

    tf_broadcaster = new tf::TransformBroadcaster();
    is_first_scan = true;
    got_new_local_map = false;
    got_new_global_map = false;

    particle_filter = new ParticleFilter(particle_count);
    particle_filter->set_prediction_params(x_x, x_y, x_r, y_x, y_y, y_r, r_x, r_y, r_r);

    ROS_INFO("Map frame:    %s", map_frame.c_str());
    ROS_INFO("Odom frame:   %s", odom_frame.c_str());
    ROS_INFO("Base frame:   %s", base_frame.c_str());
    ROS_INFO("Laser frame:  %s", laser_frame.c_str());
    ROS_INFO("Particles:    %d", particle_count);
    ROS_INFO("M per cell:   %f", m_per_cell);

    /* Initialize map to odom transform */
    map_to_odom.setOrigin(tf::Vector3(0,0,0));
    map_to_odom.setRotation(tf::createQuaternionFromRPY(0,0,0));
    
    /* Initialize odom to base transform */
    odom_to_base.setOrigin(tf::Vector3(0,0,0));
    odom_to_base.setRotation(tf::createQuaternionFromRPY(0,0,0));

    /* Initialize base to laser transform */
    tf::StampedTransform  _stamp_base_to_laser;
    try
    {
        ros::Time now = ros::Time::now();
        tf_listener.waitForTransform(base_frame, laser_frame, now, ros::Duration(3.0));
        tf_listener.lookupTransform(base_frame, laser_frame, now, _stamp_base_to_laser);
    }
    catch(tf::TransformException e)
    {
        ROS_WARN("Failed to lookup transformation from %s to %s (%s)", base_frame.c_str(), laser_frame.c_str() ,e.what());
        return false;
    }
    base_to_laser = _stamp_base_to_laser;                                    // convert stamped transform to transform
    double l_x = base_to_laser.getOrigin().x();
    double l_y = base_to_laser.getOrigin().y();
    double l_z = base_to_laser.getOrigin().z();
    ROS_INFO("%s to %s linear transform:   (%lf, %lf, %lf) ", base_frame.c_str(), laser_frame.c_str(), l_x, l_y, l_z);
    double l_roll, l_pitch, l_yaw;
    base_to_laser.getBasis().getRPY(l_roll,l_pitch,l_yaw);
    ROS_INFO("%s to %s angular transform:  (%lf, %lf, %lf) ", base_frame.c_str(), laser_frame.c_str(), l_roll, l_pitch, l_yaw);

    // Advertise
    global_map_plr = nh.advertise<nav_msgs::OccupancyGrid>("map", 1, true);
    global_map_metadata_plr = nh.advertise<nav_msgs::MapMetaData>("map_metadata", 1, true);

    local_map_plr = nh.advertise<nav_msgs::OccupancyGrid>("local_map", 1, true);
    local_map_metadata_plr = nh.advertise<nav_msgs::MapMetaData>("local_map_metadata", 1, true);

    particle_plr = pnh.advertise<geometry_msgs::PoseArray>(particle_topic, 5);
    
    fps_plr = nh.advertise<std_msgs::Float64>("fps", 5);

    // Subscribe to scan
    laser_scan_subscriber = new message_filters::Subscriber<sensor_msgs::LaserScan>(nh, laser_topic, 5);
    laser_scan_filter = new tf::MessageFilter<sensor_msgs::LaserScan>(*laser_scan_subscriber, tf_listener, odom_frame, 5);
    laser_scan_filter->registerCallback(boost::bind(&Slam::on_range_msg_callback, this, _1));

    // Publish thread: Public map to odometry transformation
    map_to_odom_plr_thread = new boost::thread(boost::bind(&Slam::map_to_odom_pl_thread, this));
    map_to_urbase_plr_thread = new boost::thread(boost::bind(&Slam::map_to_urbase_pl_thread, this));
    local_map_plr_thread = new boost::thread(boost::bind(&Slam::local_map_pl_thread, this));
    global_map_plr_thread = new boost::thread(boost::bind(&Slam::global_map_pl_thread, this));
    particle_plr_thread = new boost::thread(boost::bind(&Slam::particle_pl_thread, this));

    ROS_INFO("Start live slam");

    return true;
}

bool Slam::init_initial_state(const sensor_msgs::LaserScan::ConstPtr& scan_msg){
    // Get odom to base
    tf::StampedTransform  _stamp_odom_to_base;
    double l_roll, l_pitch, l_yaw;
    try
    {
        tf_listener.lookupTransform(odom_frame, base_frame, scan_msg->header.stamp, _stamp_odom_to_base);
    }
    catch(tf::TransformException e)
    {
        ROS_ERROR("In %s :Failed to lookup transformation from %s to %s (%s)", __func__ ,odom_frame.c_str(), base_frame.c_str() ,e.what());
        return false;
    }

    // Initialize map to odom
    map_to_odom.setOrigin(tf::Vector3(0,0,0));
    map_to_odom.setRotation(tf::createQuaternionFromRPY(0,0,0));
    
    // Find map to base and set inital pose for all particle
    tf::Transform map_to_base = map_to_odom * _stamp_odom_to_base;
    map_to_base.getBasis().getRPY(l_roll,l_pitch,l_yaw);
    particle_filter->set_initial_state(map_to_base.getOrigin().x(), map_to_base.getOrigin().y(), l_yaw);
    
    ROS_INFO("%s to %s linear transform:   (%lf, %lf, %lf) ", map_frame.c_str(), base_frame.c_str(), 
            map_to_base.getOrigin().x(), 
            map_to_base.getOrigin().y(),
            map_to_base.getOrigin().z());
    map_to_base.getBasis().getRPY(l_roll,l_pitch,l_yaw);
    ROS_INFO("%s to %s angular transform:  (%lf, %lf, %lf) ", map_frame.c_str(), base_frame.c_str(), l_roll, l_pitch, l_yaw);

    ROS_INFO("%s to %s linear transform:   (%lf, %lf, %lf) ", map_frame.c_str(), odom_frame.c_str(), 
            map_to_odom.getOrigin().x(), 
            map_to_odom.getOrigin().y(),
            map_to_odom.getOrigin().z());
    map_to_odom.getBasis().getRPY(l_roll,l_pitch,l_yaw);
    ROS_INFO("%s to %s angular transform:  (%lf, %lf, %lf) ", map_frame.c_str(), odom_frame.c_str(), l_roll, l_pitch, l_yaw);

    /* Save initial pose for future movement calculation */
    old_x = _stamp_odom_to_base.getOrigin().x();
    old_y = _stamp_odom_to_base.getOrigin().y();
    double r,p,y;
    _stamp_odom_to_base.getBasis().getRPY(r, p, y);
    old_t = y;

    return true;
}

bool Slam::init_correction_params(){
    particle_filter->set_correction_params(corr_iter_thresh_num, kernel_size_m/m_per_cell);
    return true;
}

bool Slam::init_mesurement_params(const sensor_msgs::LaserScan::ConstPtr& scan_msg){
    /* Initialize measurement params */
    laser_count = scan_msg->ranges.size();
    float *laser_angles = (float*)malloc(laser_count*sizeof(float));
    float theta = scan_msg->angle_min;
    for(unsigned int i=0; i<laser_count; ++i)
    {
        laser_angles[i]=theta;
        theta += scan_msg->angle_increment;
    }
    ROS_INFO("Laser angles in %s frame: min: %.3f max: %.3f inc: %.3f", laser_frame.c_str() ,scan_msg->angle_min, scan_msg->angle_max, scan_msg->angle_increment); 
    particle_filter->set_measurement_params(laser_count, laser_angles, min_range, max_range);
    ROS_INFO("Laser count: %d", laser_count);
    free(laser_angles);
    return true;
}

bool Slam::init_resample_params(){
    particle_filter->set_resample_params(neff_thresh);
    return true;
}

bool Slam::init_map_params(){
    particle_filter->set_map_params(initial_global_map_width,initial_global_map_height,
                                    initial_local_map_width,initial_local_map_height,
                                    true_positive, true_negative, w_thresh,
                                    m_per_cell);
    return true;
}

bool Slam::get_movement(float &dx, float &dy, float &dt, const sensor_msgs::LaserScan::ConstPtr &scan_msg){
    // Get new base link pose in odom frame
    tf::StampedTransform stamped_odom_to_base;
    try
    {
        tf_listener.lookupTransform(odom_frame, base_frame, scan_msg->header.stamp, stamped_odom_to_base);
    }
    catch(tf::TransformException e)
    {
        ROS_WARN("In %s :Failed to lookup transformation from %s to %s (%s)", __func__, odom_frame.c_str(), base_frame.c_str() ,e.what());
        return false;
    }
    odom_to_base = stamped_odom_to_base;
    float new_x, new_y, new_t;
    new_x = odom_to_base.getOrigin().x();
    new_y = odom_to_base.getOrigin().y();
    double r,p,y;
    odom_to_base.getBasis().getRPY(r, p, y);
    new_t = y;

    // Find the movement
    dx = new_x - old_x;
    dy = new_y - old_y;
    dt = new_t - old_t;
    if(dt < -M_PI) dt += 2*M_PI;
    if(dt > M_PI) dt -= 2*M_PI;

    // Save new pose to old pose
    old_x = new_x;
    old_y = new_y;
    old_t = new_t;

    return true;
}

void Slam::on_range_msg_callback(const sensor_msgs::LaserScan::ConstPtr& scan_msg){

    // start_ = ros::WallTime::now();
    auto t1 = std::chrono::high_resolution_clock::now();

    ROS_INFO_ONCE("First scan message received");

    if(scan_msg->header.frame_id != laser_frame){
        ROS_WARN("Desired to receive laser scan message in frame %s, but receive message in frame %s", laser_frame.c_str(), scan_msg->header.frame_id.c_str());
        UNEXPECTED_RETURN();
    }

    if(is_first_scan){
        ROS_INFO("First scan init");

        if(!init_initial_state(scan_msg)) UNEXPECTED_RETURN();
        if(!init_correction_params()) UNEXPECTED_RETURN();
        if(!init_mesurement_params(scan_msg)) UNEXPECTED_RETURN();
        if(!init_resample_params()) UNEXPECTED_RETURN();
        if(!init_map_params()) UNEXPECTED_RETURN();

        particle_filter->set_measurement(scan_msg->ranges.data());
        if(particle_filter->local_map_update()){
            got_new_local_map = true;
            if(particle_filter->global_map_update_force()){
                got_new_global_map = true;
                /* Turn off the flag */
                is_first_scan = false;
                ROS_INFO("First scan init: Done");
            }
            else ROS_WARN("Global map update failed");
        }
        else ROS_WARN("Local map update failed");
    }

    // Check if laser count is correct
    if(scan_msg->ranges.size() != laser_count){
        ROS_WARN("Desired to receive laser scan message has %d beams, but receive message has %d beams", laser_count, (int)scan_msg->ranges.size());
        UNEXPECTED_RETURN();
    }

    // Predcition
    float dx, dy, dt;
    if(!get_movement(dx, dy, dt, scan_msg)) UNEXPECTED_RETURN();
    particle_filter->predict(dx, dy, dt);

    // Set measurement
    particle_filter->set_measurement(scan_msg->ranges.data());

    // Correction
    particle_filter->correct();

    // Initialize map
    static int scan_cnt=0;
    if(scan_cnt < initialization_scan){
        ROS_INFO_ONCE("Initializing: Don't move.");
        if(particle_filter->global_map_update_force()){
            got_new_global_map = true;
        }
        scan_cnt++;
    }
    else if(particle_filter->score()){ // Score
        ROS_INFO_ONCE("Initializing: Done. Ready to move.");

        // Resample
        static float dx_resample_acc=0, dy_resample_acc=0, dt_resample_acc=0;
        dx_resample_acc += abs(dx); dy_resample_acc += abs(dy); dt_resample_acc += abs(dt);
        if(dx_resample_acc > dx_resample_thresh || dy_resample_acc > dy_resample_thresh || dt_resample_acc > dt_resample_thresh){
            dx_resample_acc=0; dy_resample_acc=0; dt_resample_acc=0;
            particle_filter->resample_force();
            particle_filter->score();
        }   
        else if (particle_filter->resample()){
            particle_filter->score();
        }

        if(particle_filter->local_map_update()){
            got_new_local_map = true;
        }

        static bool map_update_flag=false;
        if(abs(dx) > dx_map_update_thresh || abs(dy) > dy_map_update_thresh || abs(dt) > dt_map_update_thresh)
            map_update_flag=true;
        if(map_update_flag){
            if(particle_filter->global_map_update()){
                map_update_flag=false;
                got_new_global_map = true;
            }
        }
    }

    // end_ = ros::WallTime::now();
    // std_msgs::Float64 msg;
    // msg.data = (double)(end_ - start_).toNSec()/10e6;
    std_msgs::Float64 msg;
    auto t2 = std::chrono::high_resolution_clock::now();
    msg.data = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    fps_plr.publish(msg);

}

void Slam::map_to_odom_pl_thread(){
    ROS_INFO("[THREAD] %s to %s publish thread: running", map_frame.c_str(), odom_frame.c_str());
    ros::Rate r(1.0 / map_to_odom_pl_period);
    while(ros::ok()){
        if(is_first_scan) {
            r.sleep();
            continue;
        };

        map_to_odom_mutex.lock();
        ros::Time tf_expiration = ros::Time::now() + ros::Duration(map_to_odom_pl_period);
        
        float x,y,t;
        particle_filter->get_particle_best(&x, &y, &t);
        if(isnan(x) || isnan(y) || isnan(t)) {
            ROS_ERROR_ONCE("x || y || t is nan");
            continue;
        }
        tf::Transform map_to_base;
        map_to_base.setOrigin(tf::Vector3(x,y,0));
        map_to_base.setRotation(tf::createQuaternionFromYaw(t));
        map_to_odom = map_to_base * odom_to_base.inverse();
        tf_broadcaster->sendTransform( tf::StampedTransform(map_to_odom, tf_expiration, map_frame, odom_frame));
        map_to_odom_mutex.unlock();

        r.sleep();
        ROS_INFO_ONCE("First transformation from %s to %s has been published", map_frame.c_str(), odom_frame.c_str());
    }
}

void Slam::map_to_urbase_pl_thread(){
    ROS_INFO("[THREAD] %s to %s publish thread: running", odom_frame.c_str(), urbase_frame.c_str());
    ros::Rate r(1.0 / map_to_urbase_pl_period);
    static bool first = true;

    while(ros::ok()){
        if(is_first_scan) {
            r.sleep();
            continue;
        }

        tf::StampedTransform  stamped_map_to_base;
        try
        {
            tf_listener.lookupTransform(map_frame, base_frame, ros::Time(0), stamped_map_to_base);
        }
        catch(tf::TransformException e)
        {
            if(!first) ROS_WARN("In %s : Failed to lookup transformation from %s to %s (%s)", __func__ , map_frame.c_str(), base_frame.c_str() ,e.what());
            continue;
        }
        first = false;
        
        float x = stamped_map_to_base.getOrigin().getX();
        float y = stamped_map_to_base.getOrigin().getY();
        x = ((int)(x/m_per_cell)) * m_per_cell;
        y = ((int)(y/m_per_cell)) * m_per_cell;
        stamped_map_to_base.setOrigin(tf::Vector3(x,y,0));

        stamped_map_to_base.setRotation(tf::createQuaternionFromRPY(0,0,0));
        ros::Time tf_expiration  = ros::Time::now() + ros::Duration(map_to_urbase_pl_period);
        tf_broadcaster->sendTransform(tf::StampedTransform(stamped_map_to_base, tf_expiration, map_frame, urbase_frame));
        r.sleep();
        ROS_INFO_ONCE("First transformation from %s to %s has been published", odom_frame.c_str(), urbase_frame.c_str());
    }
}

void Slam::local_map_pl_thread(){

    ROS_INFO("[THREAD] Local map publish thread: running");
    ros::Rate r(1.0 / local_map_pl_period);

    float *local_map;
    uint32_t map_width_cell, map_height_cell;

    while(ros::ok()){
        if(is_first_scan) {
            r.sleep();
            continue;
        };


        if(local_map_plr.getNumSubscribers() == 0){
            r.sleep();
            continue; 
        }

        if(got_new_local_map){
            got_new_local_map = false;

            particle_filter->get_local_map(&map_width_cell, &map_height_cell, &m_per_cell, local_map);

            local_map_msg.map.info.width = map_width_cell;
            local_map_msg.map.info.height = map_height_cell;
            local_map_msg.map.info.resolution = m_per_cell;

            local_map_msg.map.data.resize(map_width_cell * map_height_cell);
            for(int i=0; i < map_width_cell * map_height_cell; i++){
                if(local_map[i] == 0) local_map_msg.map.data[i] = -1;
                else local_map_msg.map.data[i] = local_map[i] < 0 ? 0 : 100;
            }

            local_map_msg.map.header.stamp = ros::Time::now();
            local_map_msg.map.header.frame_id = urbase_frame;

            local_map_msg.map.info.origin.position.x = -(float)map_width_cell*m_per_cell/2.0f;
            local_map_msg.map.info.origin.position.y = -(float)map_height_cell*m_per_cell/2.0f;

            local_map_plr.publish(local_map_msg.map);
            local_map_metadata_plr.publish(local_map_msg.map.info);
        }
        
        r.sleep();
    }
}

void Slam::global_map_pl_thread(){

    ROS_INFO("[THREAD] Global map publish thread: running");
    ros::Rate r(1.0 / global_map_pl_period);

    float *global_map;
    uint32_t map_width_cell, map_height_cell;

    while(ros::ok()){
        if(is_first_scan) {
            r.sleep();
            continue;
        };

        if(global_map_plr.getNumSubscribers() == 0){
            r.sleep();
            continue; 
        }

        if(got_new_global_map){
            got_new_global_map = false;

            particle_filter->get_global_map(&map_width_cell, &map_height_cell, &m_per_cell, global_map);

            global_map_msg.map.info.width = map_width_cell;
            global_map_msg.map.info.height = map_height_cell;
            global_map_msg.map.info.resolution = m_per_cell;

            global_map_msg.map.data.resize(map_width_cell * map_height_cell);
            for(int i=0; i < map_width_cell * map_height_cell; i++){
                if(global_map[i] == 0) global_map_msg.map.data[i] = -1;
                else global_map_msg.map.data[i] = global_map[i] < 0 ? 0 : 100;
            }

            global_map_msg.map.header.stamp = ros::Time::now();
            global_map_msg.map.header.frame_id = map_frame;

            global_map_msg.map.info.origin.position.x = -(float)map_width_cell*m_per_cell/2.0f;
            global_map_msg.map.info.origin.position.y = -(float)map_height_cell*m_per_cell/2.0f;

            global_map_plr.publish(global_map_msg.map);
            global_map_metadata_plr.publish(global_map_msg.map.info);
        }

        r.sleep();
    }
}

void Slam::particle_pl_thread(){
    ROS_INFO("[THREAD] Particle publish thread: running");
    ros::Rate r(1.0 / particle_pl_period);

    float *x, *y, *t;
    int len;

    while(ros::ok()){
        if(is_first_scan) {
            r.sleep();
            continue;
        };


        if(particle_plr.getNumSubscribers() == 0){
            r.sleep();
            continue; 
        }

        partiles_msg.poses.clear();
        particle_filter->get_x(x, len);
        particle_filter->get_y(y, len);
        particle_filter->get_t(t, len);

        for(int i=0 ; i<len; i++){
            geometry_msgs::Pose pose;
            pose.position.x = x[i];
            pose.position.y = y[i];
            tf::Quaternion q;
                q.setRPY(0,0,t[i]);
            tf::quaternionTFToMsg(q, pose.orientation);
            partiles_msg.poses.push_back(pose);
        }
        partiles_msg.header.frame_id = map_frame;
        particle_plr.publish(partiles_msg);

        r.sleep();
    }
}

};
