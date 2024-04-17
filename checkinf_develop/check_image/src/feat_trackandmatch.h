#ifndef VISUAL_FEATTRACKANDMATCH_H
#define VISUAL_FEATTRACKANDMATCH_H

#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>
#include <memory>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cam_model.h"
#include "feat_database.h"



namespace Visual {

    class FeatTrack{

        public:

            FeatTrack(std::shared_ptr<CamModel> camm, std::shared_ptr<FeatureDatabase> fdb);

            void process_new_camera(const CamData &cmsg);

        protected:

            void FeatExtract(const cv::Mat &img, std::vector<cv::KeyPoint> &pts0, std::vector<size_t> &ids0);

            void FeatTrackandMatch(const cv::Mat &img0, const cv::Mat &img1, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, std::vector<uchar> &mask_out);

            void robust_ratio_test(std::vector<std::vector<cv::DMatch>> &matches);

            void robust_symmetry_test(std::vector<std::vector<cv::DMatch>> &matches1, std::vector<std::vector<cv::DMatch>> &matches2,
                                           std::vector<cv::DMatch> &good_matches);

            std::shared_ptr<CamModel> cmodel;

            std::shared_ptr<FeatureDatabase> db;

            /// Last set of images (use map so all trackers render in the same order)
            cv::Mat img_last;

            /// Last set of tracked points
            std::vector<cv::KeyPoint> pts_last;

            /// Set of IDs of each current feature in the database
            std::vector<size_t> ids_last;

            // Descriptor matrices
            cv::Mat desc_last;

            double last_frame = -1;
            std::mutex mtx;

            int grid_x = 10;
            int grid_y = 5;

            // Minimum pixel distance to be "far away enough" to be a different extracted feature
            int min_px_dist = 5;

            // How many pyramono levels to track
            int pyr_levels = 3; 
            cv::Size win_size = cv::Size(20, 20); 

            int num_features = 1000;

            /// Master ID for this tracker (atomic to allow for multi-threading)
            std::atomic<size_t> currid = 1;

            // Our orb extractor
            cv::Ptr<cv::ORB> orb0 = cv::ORB::create();
            cv::Ptr<cv::ORB> orb1 = cv::ORB::create();

            // Our descriptor matcher
            cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

            // The ratio between two kNN matches, if that ratio is larger then this threshold
            // then the two features are too close, so should be considered ambiguous/bad match
            double knn_ratio = 0.70;

            
    };
}// namespace Visual

#endif // VISUAL_FEATTRACKANDMATCH_H