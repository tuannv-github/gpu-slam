#include <ros/ros.h>
#include <Slam.h>

using namespace SLAM;

int main(int argc, char** argv){
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<Slam>());
	rclcpp::shutdown();
	return 0;
}