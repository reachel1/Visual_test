#include "feat_trackandmatch.h"

using namespace Visual;

FeatTrack::FeatTrack(std::shared_ptr<CamModel> camm, std::shared_ptr<FeatureDatabase> fdb): cmodel(camm), db(fdb){}

void FeatTrack::process_new_camera(const CamData &cmsg) {
    
    double current_frame = cmsg.timestamp;
    if(current_frame <= last_frame){
        std::cout<<"[CAMERA]Img time is messed!"<<std::endl;
        std::cout<<"current frame : "<<current_frame<<" || last frame : "<<last_frame<<std::endl;
        std::exit(EXIT_FAILURE);
    }
    last_frame = current_frame;
    cv::Mat img = cmsg.image;

    // If we didn't have any successful tracks last time, just extract this time
    // This also handles, the tracking initalization on the first call to this extractor
    if (pts_last.empty()) {
        // Detect new features
        std::vector<cv::KeyPoint> good_mono;
        std::vector<size_t> good_ids_mono;
        FeatTrack::FeatExtract(img, good_mono, good_ids_mono);
        // Save the current image and pyramono
        {
            std::lock_guard<std::mutex> lck(mtx);
            img_last = img;
            pts_last = good_mono;
            ids_last = good_ids_mono;
            for (size_t i = 0; i < good_mono.size(); i++) {
                cv::Point2f npt_l = cmodel->undistort_cv(good_mono.at(i).pt);
                db->update_feature(good_ids_mono.at(i), current_frame, good_mono.at(i).pt.x, good_mono.at(i).pt.y, npt_l.x, npt_l.y);
            }
        }
        return;
    }

    // First we should make that the last images have enough features so we can do KLT
    // This will "top-off" our number of tracks so always have a constant number
    int pts_before_detect = 0;
    std::vector<cv::KeyPoint> pts_mono_old;
    std::vector<size_t> ids_mono_old;
    {
        std::lock_guard<std::mutex> lck(mtx);
        pts_before_detect = (int)pts_last.size();
        pts_mono_old = pts_last;
        ids_mono_old = ids_last;
    }
    FeatTrack::FeatExtract(img_last, pts_mono_old, ids_mono_old);
    // std::cout<<pts_mono_old.size()<<" "<<ids_mono_old.size()<<std::endl;
    assert(pts_mono_old.size() == ids_mono_old.size());
    // Our return success masks, and predicted new features
    std::vector<uchar> mask_mm;
    std::vector<cv::KeyPoint> pts_mono_new = pts_mono_old;
    FeatTrack::FeatTrackandMatch(img_last, img, pts_mono_old, pts_mono_new, mask_mm);
    // std::cout<<pts_mono_old.size()<<" "<<pts_mono_new.size()<<" "<<ids_mono_old.size()<<std::endl;
    assert(pts_mono_new.size() == ids_mono_old.size());

    // If any of our mask is empty, that means we didn't have enough to do ransac, so just return
    if (mask_mm.empty()) {
        {
            std::lock_guard<std::mutex> lck(mtx);
            img_last = img;
            pts_last.clear();
            ids_last.clear();
        }
        std::cout<<"[CAMERA]: Failed to get enough points to track, resetting....."<<std::endl;;
        return;
    }

    // Get our "good tracks"
    std::vector<cv::KeyPoint> good_mono;
    std::vector<size_t> good_ids_mono;

    // Loop through all mono points
    for (size_t i = 0; i < pts_mono_new.size(); i++) {
        // Ensure we do not have any bad KLT tracks (i.e., points are negative)
        if (pts_mono_new.at(i).pt.x < 0 || pts_mono_new.at(i).pt.y < 0 || (int)pts_mono_new.at(i).pt.x >= img.cols ||
            (int)pts_mono_new.at(i).pt.y >= img.rows)
            continue;
        // If it is a good track, and also tracked from mono to right
        if (mask_mm[i]) {
            good_mono.push_back(pts_mono_new[i]);
            good_ids_mono.push_back(ids_mono_old[i]);
        }
    }
    std::cout<<"[CAMERA-mono]current frame track feats num "<<good_mono.size()<<std::endl;
    // Update our feature database, with theses new observations
    for (size_t i = 0; i < good_mono.size(); i++) {
        cv::Point2f npt_l = cmodel->undistort_cv(good_mono.at(i).pt);
        db->update_feature(good_ids_mono.at(i), current_frame, good_mono.at(i).pt.x, good_mono.at(i).pt.y, npt_l.x, npt_l.y);
    }

    // Move forward in time
    {
        std::lock_guard<std::mutex> lck(mtx);
        img_last = img;
        pts_last = good_mono;
        ids_last = good_ids_mono;
    }

}

void FeatTrack::FeatExtract(const cv::Mat &img, std::vector<cv::KeyPoint> &pts0, std::vector<size_t> &ids0){
    
    // void goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
    // 检测Harrisuse（HarrisDetector = true）/Shi-Tomasi角点（HarrisDetector = false）
    printf("[CAMERA]last frame feats num %d\n",pts0.size());
    // Create a 2D occupancy grid for this current image
    // Note that we scale this down, so that each grid point is equal to a set of pixels
    // This means that we will reject points that less than grid_px_size points away then existing features
    cv::Size size_close((int)((float)img.cols / (float)min_px_dist), (int)((float)img.rows / (float)min_px_dist)); // width x height
    cv::Mat grid_2d_close = cv::Mat::zeros(size_close, CV_8UC1);
    float size_x = (float)img.cols / (float)grid_x;
    float size_y = (float)img.rows / (float)grid_y;
    cv::Size size_grid(grid_x, grid_y); // width x height
    cv::Mat grid_2d_grid = cv::Mat::zeros(size_grid, CV_8UC1);
    auto it0 = pts0.begin();
    auto it1 = ids0.begin();
    while (it0 != pts0.end()) {
        // Get current keypoint, check that it is in bounds
        cv::KeyPoint kpt = *it0;
        int x = (int)kpt.pt.x;
        int y = (int)kpt.pt.y;
        int edge = 10;
        if (x < edge || x >= img.cols - edge || y < edge || y >= img.rows - edge) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate occupancy coordinates for close points
        int x_close = (int)(kpt.pt.x / (float)min_px_dist);
        int y_close = (int)(kpt.pt.y / (float)min_px_dist);
        if (x_close < 0 || x_close >= size_close.width || y_close < 0 || y_close >= size_close.height) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Calculate what grid cell this feature is in
        int x_grid = std::floor(kpt.pt.x / size_x);
        int y_grid = std::floor(kpt.pt.y / size_y);
        if (x_grid < 0 || x_grid >= size_grid.width || y_grid < 0 || y_grid >= size_grid.height) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
            continue;
        }
        // Check if this keypoint is near another point
        if (grid_2d_close.at<uint8_t>(y_close, x_close) > 127) {
            it0 = pts0.erase(it0);
            it1 = ids0.erase(it1);
        continue;
        }
        // Else we are good, move forward to the next point
        grid_2d_close.at<uint8_t>(y_close, x_close) = 255;
        if (grid_2d_grid.at<uint8_t>(y_grid, x_grid) < 255) {
            grid_2d_grid.at<uint8_t>(y_grid, x_grid) += 1;
        }
        it0++;
        it1++;
    }

    // First compute how many more features we need to extract from this image
    // If we don't need any features, just return
    double min_feat_percent = 0.50;
    int num_featsneeded = num_features - (int)pts0.size();
    if (num_featsneeded < std::min(20, (int)(min_feat_percent * num_features)))
        return;

    // Extract new feats
    std::vector<cv::Point2f> pts0_ext;
    cv::goodFeaturesToTrack(img, pts0_ext, num_features, 0.01, 3.0, cv::Mat(), 3, false, 0.04);
    // Sub-pixel refine
    cv::cornerSubPix(img, pts0_ext, cv::Size(5, 5), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, 0.001));

    // Now, reject features that are close a current feature and add new feats to pts in current frame
    for (auto &pt : pts0_ext) {
        // Check that it is in bounds
        int x_grid = (int)(pt.x / (float)min_px_dist);
        int y_grid = (int)(pt.y / (float)min_px_dist);
        if (x_grid < 0 || x_grid >= size_close.width || y_grid < 0 || y_grid >= size_close.height)
            continue;
        // See if there is a point at this location
        if (grid_2d_close.at<uint8_t>(y_grid, x_grid) > 127)
            continue;
        // Else lets add it!
        cv::KeyPoint kpt;
        kpt.pt = pt;
        grid_2d_close.at<uint8_t>(y_grid, x_grid) = 255;
        pts0.push_back(kpt);
        // move id foward and append this new point
        size_t temp = ++currid;
        ids0.push_back(temp);
    }
    printf("[CAMERA]feats num after add %d\n",pts0.size());
}


void FeatTrack::FeatTrackandMatch(const cv::Mat &img0, const cv::Mat &img1, std::vector<cv::KeyPoint> &kpts0,
                                std::vector<cv::KeyPoint> &kpts1, std::vector<uchar> &mask_out){

    // We must have equal vectors
    assert(kpts0.size() == kpts1.size());

    // Return if we don't have any points
    if (kpts0.empty() || kpts1.empty())
        return;

    std::vector<cv::Mat> imgpyr0, imgpyr1;
    cv::buildOpticalFlowPyramid(img0, imgpyr0, win_size, pyr_levels);
    cv::buildOpticalFlowPyramid(img1, imgpyr1, win_size, pyr_levels);

    // Convert keypoints into points (stupid opencv stuff)
    std::vector<cv::Point2f> pts0, pts1, pts0_back;
    for (size_t i = 0; i < kpts0.size(); i++) {
        pts0.push_back(kpts0.at(i).pt);
        pts0_back.push_back(kpts0.at(i).pt);
        pts1.push_back(kpts1.at(i).pt);
    }

    // Now do KLT tracking to get the valid new points
    std::vector<uchar> mask_klt, mask_klt_back;
    std::vector<float> error, error_back;
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 40, 0.001);

    // Do LK forward and back
    cv::calcOpticalFlowPyrLK(imgpyr0, imgpyr1, pts0, pts1, mask_klt, error, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
    cv::calcOpticalFlowPyrLK(imgpyr1, imgpyr0, pts1, pts0_back, mask_klt_back, error_back, win_size, pyr_levels, term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);

    // Copy back the updated positions
    for (size_t i = 0; i < pts0.size(); i++) {
        kpts0.at(i).pt = pts0.at(i);
        kpts1.at(i).pt = pts1.at(i);
    }
    
    // // Do BRIEF match
    // cv::Mat desc0, desc1;
    // orb0->compute(img0, kpts0, desc0);
    // orb1->compute(img1, kpts1, desc1);
    
    // // Our 1to2 and 2to1 match vectors
    // std::vector<std::vector<cv::DMatch>> matches0to1, matches1to0;

    // // Match descriptors (return 2 nearest neighbours)
    // matcher->knnMatch(desc0, desc1, matches0to1, 2);
    // matcher->knnMatch(desc1, desc0, matches1to0, 2);

    // // Do a ratio test for both matches
    // FeatTrack::robust_ratio_test(matches0to1);
    // FeatTrack::robust_ratio_test(matches1to0);

    // // Finally do a symmetry test
    // std::vector<cv::DMatch> matches_good;
    // robust_symmetry_test(matches0to1, matches1to0, matches_good);

    // Loop through and record only ones that are valid
    for (size_t i = 0; i < mask_klt.size(); i++) {
        // Loop through all left matches, and find the old "train" id
        // int seq = -1;
        // for (size_t j = 0; j < matches_good.size(); j++) {
        //     if (matches_good[j].trainIdx == (int)i) {
        //         seq = matches_good[j].queryIdx;
        //     }
        // }
        auto mask = (uchar)((i < mask_klt.size() && mask_klt[i] && i < mask_klt_back.size() && mask_klt_back[i]
                && std::floor(pts0_back.at(i).x) == std::floor(pts0.at(i).x)  
                && std::floor(pts0_back.at(i).y) == std::floor(pts0.at(i).y) 
                //&& seq == i
                ) ? 1 : 0);
        mask_out.push_back(mask);
    }

}

void FeatTrack::robust_ratio_test(std::vector<std::vector<cv::DMatch>> &matches) {
  // Loop through all matches
  for (auto &match : matches) {
    // If 2 NN has been identified, else remove this feature
    if (match.size() > 1) {
      // check distance ratio, remove it if the ratio is larger
      if (match[0].distance / match[1].distance > knn_ratio) {
        match.clear();
      }
    } else {
      // does not have 2 neighbours, so remove it
      match.clear();
    }
  }
}

void FeatTrack::robust_symmetry_test(std::vector<std::vector<cv::DMatch>> &matches1, std::vector<std::vector<cv::DMatch>> &matches2,
                                           std::vector<cv::DMatch> &good_matches) {
  // for all matches image 1 -> image 2
  for (auto &match1 : matches1) {
    // ignore deleted matches
    if (match1.empty() || match1.size() < 2)
      continue;
    // for all matches image 2 -> image 1
    for (auto &match2 : matches2) {
      // ignore deleted matches
      if (match2.empty() || match2.size() < 2)
        continue;
      // Match symmetry test
      if (match1[0].queryIdx == match2[0].trainIdx && match2[0].queryIdx == match1[0].trainIdx) {
        // add symmetrical match
        good_matches.emplace_back(cv::DMatch(match1[0].queryIdx, match1[0].trainIdx, match1[0].distance));
        // next match in image 1 -> image 2
        break;
      }
    }
  }
}
