#include <ros/ros.h>
#include <Slam.h>

using namespace SLAM;

int main(int argc, char** argv){
	// if(ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug)){
	//     ros::console::notifyLoggerLevelsChanged();
	// }
	setbuf(stdout, NULL);
	
	ros::init(argc,argv,"slam");

	Slam slam;
	while(slam.start_live_slam()!=true);
	ros::spin();

	return 0;
}