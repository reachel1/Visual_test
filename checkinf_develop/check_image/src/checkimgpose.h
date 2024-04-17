#ifndef VISUAL_CHECKIMGPOSE_H
#define VISUAL_CHECKIMGPOSE_H

#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>
#include <memory>
#include <string>
#include <shared_mutex>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "feat_trackandmatch.h"
#include "data_and_cal.h"
#include "optimize.h"

#define   M_PI   3.14159265358979323846  /* pi */

namespace Visual {

class CheckImgPose{

public:

  CheckImgPose();

  void feed_img(const sensor_msgs::Image::ConstPtr &msg);

  cv::Mat get_track();

  void load_gt(const geometry_msgs::PoseStamped::ConstPtr &msg){

    Eigen::Matrix4d T_gtpose = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q(msg->pose.orientation.w,msg->pose.orientation.x,msg->pose.orientation.y,msg->pose.orientation.z);
    Eigen::Matrix3d R_gtpose = q.normalized().toRotationMatrix();
    Eigen::Vector3d t_gtpose(msg->pose.position.x,msg->pose.position.y,msg->pose.position.z);
    T_gtpose.block(0,0,3,3) = R_gtpose;
    T_gtpose.block(0,3,3,1) = t_gtpose;
    double timestamp = msg->header.stamp.toSec();
    Eigen::Matrix4d T_gtpose_caml = T_gtpose * calib_CltoB;
    gt_cpose_gtime.insert(std::pair<double, Eigen::Matrix4d>(timestamp, T_gtpose_caml));

  }

  void output_pose(){
    double begin_time = so3_relative.begin()->first;
    if(gt_cpose_ctime.find(begin_time) == gt_cpose_ctime.end())return;
    Eigen::Matrix3d begin_ori = gt_cpose_ctime.find(begin_time)->second.block(0,0,3,3);
    Eigen::Matrix3d ori = begin_ori;
    Eigen::Vector3d eular_ori = ori.eulerAngles(0,1,2);
    printf("Eular in %.4f: %.3f.%.3f,%.3f(esi) || %.3f,%.3f,%.3f(gt)\n", begin_time, eular_ori(0),eular_ori(1),
                    eular_ori(2),eular_ori(0),eular_ori(1),eular_ori(2));
    for(auto &rep:so3_relative){
      Eigen::Vector3d ori_ctime = rep.second;
      if(gt_cpose_ctime.find(rep.first) == gt_cpose_ctime.end()){
          printf("\033[47;31mError gtcimg get !!!\033[0m");
          std::exit(EXIT_FAILURE);
      }
      
      auto time = ++gt_cpose_ctime.find(rep.first);
      Eigen::Matrix3d ori_rel = Visual::exp_so3(ori_ctime).transpose();
      ori = ori * ori_rel;
      Eigen::Vector3d eular_ori_esi = ori.eulerAngles(0,1,2);
      Eigen::Matrix3d R_ori_gt = time->second.block(0,0,3,3);
      Eigen::Vector3d eular_ori_gt = R_ori_gt.eulerAngles(0,1,2);
      printf("Eular in %.4f: %.3f.%.3f,%.3f(esi) || %.3f,%.3f,%.3f(gt)\n", time->first, eular_ori_esi(0)/M_PI*180,eular_ori_esi(1)/M_PI*180,
                    eular_ori_esi(2)/M_PI*180,eular_ori_gt(0)/M_PI*180,eular_ori_gt(1)/M_PI*180,eular_ori_gt(2)/M_PI*180);
    }
  }

  

protected:

  void optimize_pose();

  void gtinterpole_time_to_cam(double cam_time);

  std::shared_ptr<FeatTrack> ft;

  std::shared_ptr<CamModel> camm;

  std::shared_ptr<FeatureDatabase> fdb;

  Visual::CamData current_frame;

  Visual::CamData last_frame;

  std::deque<double> time_seq;

  std::map<double, Eigen::Matrix4d> gt_cpose_gtime;
  std::map<double, Eigen::Matrix4d> gt_cpose_ctime;
  double last_gtime = -1;

  Eigen::Matrix4d calib_CltoB;

  std::map<double, Eigen::Vector3d> so3_relative;



};

} // namespace Visual

#endif // VISUAL_CHECKIMGPOSE_H
