#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <ros/callback_queue.h>
#include <iostream>
#include <sstream> 
#include <fstream>
#include <stdlib.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "checkimgpose.h"


int main(int argc, char **argv) {

    // Launch our ros node
    ros::init(argc, argv, "run_check_image");
    auto nh = std::make_shared<ros::NodeHandle>("~");

    ROS_INFO("check image begin...");

    // Get params
    if(argc < 4){
        ROS_INFO("Please input --- rosrun check_image run_check_image bag_path cam_topic gt_topic");
        std::exit(EXIT_FAILURE);
    }
    std::string bag_path, cam_topic, gt_topic;
    bag_path = argv[1];
    cam_topic = argv[2];
    gt_topic = argv[3];
    std::cout<<bag_path<<" : "<<cam_topic<<std::endl;
    std::cout<<bag_path<<" : "<<gt_topic<<std::endl;
    // Open the bag
    rosbag::Bag bag;
    bag.open(bag_path,rosbag::bagmode::Read);
    rosbag::View view_bag;
    view_bag.addQuery(bag);

    // New check
    std::shared_ptr<Visual::CheckImgPose> cio;
    cio = std::make_shared<Visual::CheckImgPose>();

    // View gt in bag
    std::shared_ptr<rosbag::View> view_gt;
    rosbag::View::iterator view_gt_iter;
    geometry_msgs::PoseStamped::ConstPtr msg_gt;
    view_gt = std::make_shared<rosbag::View>(bag, rosbag::TopicQuery(gt_topic),view_bag.getBeginTime(),view_bag.getEndTime());
    view_gt_iter = view_gt->begin();
    while(ros::ok()){
        if(view_gt_iter == view_gt->end())break;
        msg_gt = view_gt_iter->instantiate<geometry_msgs::PoseStamped>();
        cio->load_gt(msg_gt);
        view_gt_iter++;
    }

    // View img in bag
    std::shared_ptr<rosbag::View> view_cam;
    rosbag::View::iterator view_cam_iter;
    sensor_msgs::Image::ConstPtr msg_img;
    view_cam = std::make_shared<rosbag::View>(bag, rosbag::TopicQuery(cam_topic), view_bag.getBeginTime(),view_bag.getEndTime());
    view_cam_iter = view_cam->begin();
    // Create image transport
    ros::Publisher it_pub_tracks;
    it_pub_tracks = nh->advertise<sensor_msgs::Image>("/track", 2);
    ROS_INFO("Publishing: %s\n", it_pub_tracks.getTopic().c_str());
    int seq = 0;
    while(ros::ok()){
        if(view_cam_iter == view_cam->end())break;
        msg_img = view_cam_iter->instantiate<sensor_msgs::Image>();
        if(msg_img->header.stamp.toSec() > 1559193281.756724)break;
        cio->feed_img(msg_img); 
        cv::Mat img_track = cio->get_track();
        // Create our message
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        sensor_msgs::ImagePtr track_msg = cv_bridge::CvImage(header, "bgr8", img_track).toImageMsg();
        // Publish
        it_pub_tracks.publish(track_msg);

        view_cam_iter++;
        seq++;
    }
    cio->output_pose();
    
    // Done!
    return EXIT_SUCCESS;

}