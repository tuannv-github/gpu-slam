#ifndef __COMMON_CUH__
#define __COMMON_CUH__

namespace SLAM{

typedef enum {
    VVLD,
    VIVL,
    VINF
}valid_t;

#define UNEXPECTED_RETURN()   {\
    ROS_INFO("Unexpected return from file %s, line %d", __FILE__, __LINE__); \
    return; \
}

};

#endif