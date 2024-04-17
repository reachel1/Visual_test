#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include "checkimgpose.h"

using namespace Visual;
using namespace cv;
using namespace std;


CheckImgPose::CheckImgPose(){
    camm = std::make_shared<CamModel>(1280, 560);
    Eigen::VectorXd cam_calib = Eigen::VectorXd::Zero(8);
    cam_calib<<8.1690378992770002e+02,8.1156803828490001e+02,6.0850726281690004e+02,2.6347599764440002e+02,
               -5.6143027800000002e-02,1.3952563200000001e-01,-1.2155906999999999e-03,-9.7281389999999998e-04;
    camm->set_value(cam_calib);
    fdb = std::make_shared<FeatureDatabase>();
    ft = std::make_shared<FeatTrack>(camm,fdb);

    // set calib
    Eigen::Matrix3d Rbtocl;
    Rbtocl<<-0.00680499,-0.0153215,0.99985,-0.999977,0.000334627,-0.00680066,-0.000230383,-0.999883,-0.0153234;
    Eigen::Vector3d pbincl;
    pbincl<<1.64239,0.247401,1.58411;
    calib_CltoB = Eigen::Matrix4d::Identity();
    calib_CltoB.block(0,0,3,3) = Rbtocl.transpose();
    calib_CltoB.block(0,3,3,1) = -Rbtocl.transpose() * pbincl;
}

cv::Mat CheckImgPose::get_track(){

    // Get the image
    // Create the measurement
    double time = current_frame.timestamp;
    cv::Mat img = current_frame.image.clone();
    cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);

    printf("output track img time %.3f\n",time);

    std::vector<Visual::Feature> featsintrack,ptslast,ptsnew;
    featsintrack = fdb->features_contain1time(time);
    for(size_t i = 0;i<featsintrack.size();i++){
        if(featsintrack.at(i).timestamps.size()>1)ptslast.push_back(featsintrack.at(i));
        else ptsnew.push_back(featsintrack.at(i));
    }
    assert(featsintrack.size() ==  (ptslast.size() + ptsnew.size()));
    auto itpto = ptslast.begin();
    while(itpto != ptslast.end()){
        cv::Point2f pt_c((*(*itpto).uvs.rbegin())(0),(*(*itpto).uvs.rbegin())(1));
        cv::circle(img, pt_c, 4, cv::Scalar(0,255,0), cv::FILLED);
        itpto++;
    }

    auto itptn = ptsnew.begin();
    while(itptn != ptsnew.end()){
        cv::Point2f pt_c((*(*itptn).uvs.rbegin())(0),(*(*itptn).uvs.rbegin())(1));
        cv::circle(img, pt_c, 4, cv::Scalar(255,0,0), cv::FILLED);
        itptn++;
    }

    cv::putText(img, "Track : "+std::to_string(featsintrack.size()), cv::Point(30, 60), 
                cv::FONT_HERSHEY_COMPLEX_SMALL, 3.0, cv::Scalar(0,0,255), 3);

    return img;
}

void CheckImgPose::feed_img(const sensor_msgs::Image::ConstPtr &msg){

    // Get the image
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
        std::cout<<"cv_bridge exception: "<<e.what()<<std::endl;;
        return;
    }

    // Create the measurement
    CamData cmsg;
    cmsg.timestamp = cv_ptr->header.stamp.toSec();
    cmsg.image = cv_ptr->image.clone();
    current_frame = cmsg;
    // append it to our queue of images
    ft->process_new_camera(cmsg);
    CheckImgPose::gtinterpole_time_to_cam(cmsg.timestamp);
    if(msg->header.stamp.toSec() < 1559193281.756724 && msg->header.stamp.toSec() >= 1559193247.373893)CheckImgPose::optimize_pose();
    last_frame = cmsg;
}


void CheckImgPose::optimize_pose(){

    double time = current_frame.timestamp;
    time_seq.push_back(time);

    if(time_seq.size() > 5)time_seq.pop_front();
    if(time_seq.size() < 2)return;

    double so3insw[time_seq.size()-1][3];
    for(int k = 0;k < time_seq.size()-1;k++){
        double time = time_seq.at(k);
        if(so3_relative.find(time) != so3_relative.end()){
            Eigen::Vector3d so3R = so3_relative[time];
            so3insw[k][0] = so3R(0);
            so3insw[k][1] = so3R(1);
            so3insw[k][2] = so3R(2);
        }else{
            so3insw[k][0] = 1e-12;
            so3insw[k][1] = 1e-12;
            so3insw[k][2] = 1e-12;
        }
        
    }
    
    ceres::Problem problem;

    // add old time relative E(R-SO3) to optimizer
    for(int i = 0;i < time_seq.size()-1;i++){
        for(int j = i+1;j < time_seq.size();j++){

            double time1 = time_seq.at(i);
            double time2 = time_seq.at(j);

            if(gt_cpose_ctime.find(time1) == gt_cpose_ctime.end() || gt_cpose_ctime.find(time2) == gt_cpose_ctime.end()){
                printf("\033[47;31mError img time !!!\033[0m");
                std::exit(EXIT_FAILURE);
            }
            Eigen::Matrix4d T1 = gt_cpose_ctime.find(time1)->second;
            Eigen::Matrix4d T2 = gt_cpose_ctime.find(time2)->second;
            Eigen::Vector3d t_rel = (Visual::Inv_se3(T2) * T1).block(0,3,3,1);
            
            std::vector<Visual::Feature> feats_has2time;
            feats_has2time = fdb->features_contain2time(time1,time2);
            if(feats_has2time.size() < 8){
                printf("Dont have enough feats to optimize (%d pair feat) in time1(%.3f) and time2(%.3f)\n",feats_has2time.size(), time1,time2);
                continue;
            }
            for(int m = 0;m < feats_has2time.size();m++){  
                int dis1 = std::distance(feats_has2time.at(m).timestamps.begin(), 
                            std::find(feats_has2time.at(m).timestamps.begin(), feats_has2time.at(m).timestamps.end(), time1));
                int dis2 = std::distance(feats_has2time.at(m).timestamps.begin(), 
                            std::find(feats_has2time.at(m).timestamps.begin(), feats_has2time.at(m).timestamps.end(), time2));
                Eigen::Vector3d xy1(feats_has2time.at(m).uvs_norm.at(dis1)(0), feats_has2time.at(m).uvs_norm.at(dis1)(1), 1), 
                            xy2(feats_has2time.at(m).uvs_norm.at(dis2)(0), feats_has2time.at(m).uvs_norm.at(dis2)(1), 1);
                ceres::CostFunction* cost_function;
                ceres::LossFunction* loss_function;
                switch(j-i){
                    case 1:
                        cost_function = new ceres::NumericDiffCostFunction<EpipolarCostFunctor1, ceres::CENTRAL, 1, 3>(new EpipolarCostFunctor1(xy1, xy2,t_rel));
                        loss_function = new ceres::HuberLoss(0.1);
                        problem.AddResidualBlock(cost_function, loss_function,  so3insw[i]);
                        break;
                    case 2:
                        cost_function = new ceres::NumericDiffCostFunction<EpipolarCostFunctor2, ceres::CENTRAL, 1, 3, 3>(new EpipolarCostFunctor2(xy1, xy2,t_rel));
                        loss_function = new ceres::HuberLoss(0.1);
                        problem.AddResidualBlock(cost_function, loss_function,  so3insw[i], so3insw[i+1]);
                        break;
                    case 3:
                        cost_function = new ceres::NumericDiffCostFunction<EpipolarCostFunctor3, ceres::CENTRAL, 1, 3, 3, 3>(new EpipolarCostFunctor3(xy1, xy2,t_rel));
                        loss_function = new ceres::HuberLoss(0.1);
                        problem.AddResidualBlock(cost_function, loss_function,  so3insw[i], so3insw[i+1], so3insw[i+2]);
                        break;
                    case 4:
                        cost_function = new ceres::NumericDiffCostFunction<EpipolarCostFunctor4, ceres::CENTRAL, 1, 3, 3, 3, 3>(new EpipolarCostFunctor4(xy1, xy2,t_rel));
                        loss_function = new ceres::HuberLoss(0.1);
                        problem.AddResidualBlock(cost_function, loss_function,  so3insw[i], so3insw[i+1], so3insw[i+2], so3insw[i+3]);
                        break;
                }
            }
        }
    }
    
    // 配置求解器
    ceres::Solver::Options options; // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR; // 增量方程如何求解
    options.minimizer_progress_to_stdout = true; // 输出到cout
    ceres::Solver::Summary summary; //
    ceres::Solve (options, &problem, &summary); // 开始优化
    std::cout << summary.BriefReport() <<std::endl;


    // save optimize results
    for(int i = 0;i < time_seq.size()-1;i++){
        double time = time_seq.at(i);
        Eigen::Vector3d newso3(so3insw[i][0],so3insw[i][1],so3insw[i][2]);
        so3_relative[time] = newso3;
    }
    
}


void CheckImgPose::gtinterpole_time_to_cam(double cam_time){

    if(last_gtime < 0){
      double first_time = gt_cpose_gtime.begin()->first;
      if(fabs(first_time - cam_time) > 5){
        printf("\033[47;31mError gt/img input !!!\033[0m");
        std::exit(EXIT_FAILURE);
      }
      if(first_time > cam_time){
        Eigen::Matrix4d T_campose = gt_cpose_gtime.begin()->second;
        gt_cpose_ctime.insert(std::pair<double, Eigen::Matrix4d>(cam_time, T_campose));
        return;
      }
      else{
        std::map<double, Eigen::Matrix4d>::iterator itc1 = gt_cpose_gtime.begin();
        std::map<double, Eigen::Matrix4d>::iterator itc2 = ++itc1;
        itc1--;
        while(itc2->first <= cam_time && itc2 != gt_cpose_gtime.end()){
          itc1++;
          itc2 = ++itc1;
          itc1--;
        }
        Eigen::Matrix4d T_inter = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d RctoG1, RctoG2;
        RctoG1 = itc1->second.block(0,0,3,3);
        RctoG2 = itc2->second.block(0,0,3,3);
        Eigen::Vector3d eulerAngle1 = RctoG1.eulerAngles(2,1,0);//RPY
        Eigen::Vector3d eulerAngle2 = RctoG2.eulerAngles(2,1,0);//RPY
        Eigen::Vector3d pcinG1 = itc1->second.block(0,3,3,1);
        Eigen::Vector3d pcinG2 = itc2->second.block(0,3,3,1);
        double ratio = (cam_time - itc1->first)/(itc2->first - itc1->first);
        Eigen::Vector3d eularAngle_inter = eulerAngle1 * (1-ratio) + eulerAngle2 * ratio;
        Eigen::Vector3d pcinG_inter = pcinG1 * (1-ratio) + pcinG2 * ratio;
        Eigen::Matrix3d RctoG_inter;
        RctoG_inter = Eigen::AngleAxisd(eularAngle_inter[0], Eigen::Vector3d::UnitZ()) * 
                      Eigen::AngleAxisd(eularAngle_inter[1], Eigen::Vector3d::UnitY()) * 
                      Eigen::AngleAxisd(eularAngle_inter[2], Eigen::Vector3d::UnitX());
        T_inter.block(0,0,3,3) = RctoG_inter;                         
        T_inter.block(0,3,3,1) = pcinG_inter;
        gt_cpose_ctime.insert(std::pair<double, Eigen::Matrix4d>(cam_time, T_inter));
        last_gtime = itc1->first; 
      }
    }
    else{
        std::map<double, Eigen::Matrix4d>::iterator itc1 = gt_cpose_gtime.find(last_gtime);
        std::map<double, Eigen::Matrix4d>::iterator itc2 = ++itc1;
        itc1--;
        while(itc2->first <= cam_time && itc2 != gt_cpose_gtime.end()){
          itc1++;
          itc2 = ++itc1;
          itc1--;
        }
        if(itc2 != gt_cpose_gtime.end()){
          Eigen::Matrix4d T_inter = Eigen::Matrix4d::Identity();
          Eigen::Matrix3d RctoG1, RctoG2;
          RctoG1 = itc1->second.block(0,0,3,3);
          RctoG2 = itc2->second.block(0,0,3,3);
          Eigen::Vector3d eulerAngle1 = RctoG1.eulerAngles(2,1,0);//RPY
          Eigen::Vector3d eulerAngle2 = RctoG2.eulerAngles(2,1,0);//RPY
          Eigen::Vector3d pcinG1 = itc1->second.block(0,3,3,1);
          Eigen::Vector3d pcinG2 = itc2->second.block(0,3,3,1);
          double ratio = (cam_time - itc1->first)/(itc2->first - itc1->first);
          Eigen::Vector3d eularAngle_inter = eulerAngle1 * (1-ratio) + eulerAngle2 * ratio;
          Eigen::Vector3d pcinG_inter = pcinG1 * (1-ratio) + pcinG2 * ratio;
          Eigen::Matrix3d RctoG_inter;
          RctoG_inter = Eigen::AngleAxisd(eularAngle_inter[0], Eigen::Vector3d::UnitZ()) * 
                        Eigen::AngleAxisd(eularAngle_inter[1], Eigen::Vector3d::UnitY()) * 
                        Eigen::AngleAxisd(eularAngle_inter[2], Eigen::Vector3d::UnitX());
          T_inter.block(0,0,3,3) = RctoG_inter; 
          T_inter.block(0,3,3,1) = pcinG_inter;
          gt_cpose_ctime.insert(std::pair<double, Eigen::Matrix4d>(cam_time, T_inter));
          last_gtime = itc1->first;
        }
        else if(itc2 == gt_cpose_gtime.end() && fabs(gt_cpose_gtime.rbegin()->first - cam_time) < 3){
          Eigen::Matrix4d T_campose = gt_cpose_gtime.rbegin()->second;
          gt_cpose_ctime.insert(std::pair<double, Eigen::Matrix4d>(cam_time, T_campose));
          return;
        }
        else{
          printf("\033[47;31mError gt/img input !!!\033[0m");
          std::exit(EXIT_FAILURE);
        }
    }

    // for(auto &pose:gt_cpose_ctime){
    //     printf("ctime: %.3f :",pose.first);
    //     std::cout<<pose.second.block(0,3,3,1).transpose()<<std::endl;
    // }
  }
