#ifndef VISUAL_CAMMODEL_H
#define VISUAL_CAMMODEL_H

#include <Eigen/Eigen>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <memory>

#include <opencv2/opencv.hpp>

namespace Visual {

/**
 * @brief Base pinhole camera model class(radtan)
 */

class CamModel {

public:
  /**
   * @brief Default constructor
   * @param width Width of the camera (raw pixels)
   * @param height Height of the camera (raw pixels)
   */
  CamModel(int width, int height) : _width(width), _height(height) {}

  /**
   * @brief This will set and update the camera calibration values.
   * This should be called on startup for each camera and after update!
   * @param calib Camera calibration information (f_x & f_y & c_x & c_y & k_1 & k_2 & k_3 & k_4)
   */
  void set_value(const Eigen::MatrixXd &calib) {

    std::lock_guard<std::mutex> lck(mtx);

    // Assert we are of size eight
    assert(calib.rows() == 8);
    camera_values = calib;

    // Camera matrix
    cv::Matx33d tempK;
    tempK(0, 0) = calib(0);
    tempK(0, 1) = 0;
    tempK(0, 2) = calib(2);
    tempK(1, 0) = 0;
    tempK(1, 1) = calib(1);
    tempK(1, 2) = calib(3);
    tempK(2, 0) = 0;
    tempK(2, 1) = 0;
    tempK(2, 2) = 1;
    camera_k_OPENCV = tempK;

    // Distortion parameters
    cv::Vec4d tempD;
    tempD(0) = calib(4);
    tempD(1) = calib(5);
    tempD(2) = calib(6);
    tempD(3) = calib(7);
    camera_d_OPENCV = tempD;
  }

  /**
   * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
   * @param uv_dist Raw uv coordinate we wish to undistort
   * @return 2d vector of normalized coordinates
   */
  Eigen::Vector2f undistort_f(const Eigen::Vector2f &uv_dist) {

    std::lock_guard<std::mutex> lck(mtx);
    // Determine what camera parameters we should use
    cv::Matx33d camK = camera_k_OPENCV;
    cv::Vec4d camD = camera_d_OPENCV;

    // Convert to opencv format
    cv::Mat mat(1, 2, CV_32F);
    mat.at<float>(0, 0) = uv_dist(0);
    mat.at<float>(0, 1) = uv_dist(1);
    mat = mat.reshape(2); // Nx1, 2-channel

    // Undistort it!
    cv::undistortPoints(mat, mat, camK, camD);

    // Construct our return vector
    Eigen::Vector2f pt_out;
    mat = mat.reshape(1); // Nx2, 1-channel
    pt_out(0) = mat.at<float>(0, 0);
    pt_out(1) = mat.at<float>(0, 1);
    return pt_out;
  }

  /**
   * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
   * @param uv_dist Raw uv coordinate we wish to undistort
   * @return 2d vector of normalized coordinates
   */
  Eigen::Vector2d undistort_d(const Eigen::Vector2d &uv_dist) {
    Eigen::Vector2f ept1, ept2;
    ept1 = uv_dist.cast<float>();
    ept2 = undistort_f(ept1);
    return ept2.cast<double>();
  }

  /**
   * @brief Given a raw uv point, this will undistort it based on the camera matrices into normalized camera coords.
   * @param uv_dist Raw uv coordinate we wish to undistort
   * @return 2d vector of normalized coordinates
   */
  cv::Point2f undistort_cv(const cv::Point2f &uv_dist) {
    Eigen::Vector2f ept1, ept2;
    ept1 << uv_dist.x, uv_dist.y;
    ept2 = undistort_f(ept1);
    cv::Point2f pt_out;
    pt_out.x = ept2(0);
    pt_out.y = ept2(1);
    return pt_out;
  }

  /**
   * @brief Given a normalized uv coordinate this will distort it to the raw image plane
   * @param uv_norm Normalized coordinates we wish to distort
   * @return 2d vector of raw uv coordinate
   */
  Eigen::Vector2f distort_f(const Eigen::Vector2f &uv_norm){

    std::lock_guard<std::mutex> lck(mtx);
    // Get our camera parameters
    Eigen::MatrixXd cam_d = camera_values;

    // Calculate distorted coordinates for radial
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double r_2 = r * r;
    double r_4 = r_2 * r_2;
    double x1 = uv_norm(0) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + 2 * cam_d(6) * uv_norm(0) * uv_norm(1) +
                cam_d(7) * (r_2 + 2 * uv_norm(0) * uv_norm(0));
    double y1 = uv_norm(1) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + cam_d(6) * (r_2 + 2 * uv_norm(1) * uv_norm(1)) +
                2 * cam_d(7) * uv_norm(0) * uv_norm(1);

    // Return the distorted point
    Eigen::Vector2f uv_dist;
    uv_dist(0) = (float)(cam_d(0) * x1 + cam_d(2));
    uv_dist(1) = (float)(cam_d(1) * y1 + cam_d(3));
    return uv_dist;
  }

  /**
   * @brief Given a normalized uv coordinate this will distort it to the raw image plane
   * @param uv_norm Normalized coordinates we wish to distort
   * @return 2d vector of raw uv coordinate
   */
  Eigen::Vector2d distort_d(const Eigen::Vector2d &uv_norm) {
    Eigen::Vector2f ept1, ept2;
    ept1 = uv_norm.cast<float>();
    ept2 = distort_f(ept1);
    return ept2.cast<double>();
  }

  /**
   * @brief Given a normalized uv coordinate this will distort it to the raw image plane
   * @param uv_norm Normalized coordinates we wish to distort
   * @return 2d vector of raw uv coordinate
   */
  cv::Point2f distort_cv(const cv::Point2f &uv_norm) {
    Eigen::Vector2f ept1, ept2;
    ept1 << uv_norm.x, uv_norm.y;
    ept2 = distort_f(ept1);
    cv::Point2f pt_out;
    pt_out.x = ept2(0);
    pt_out.y = ept2(1);
    return pt_out;
  }

  /**
   * @brief Computes the derivative of raw distorted to normalized coordinate.
   * @param uv_norm Normalized coordinates we wish to distort
   * @param H_dz_dzn Derivative of measurement z in respect to normalized
   * @param H_dz_dzeta Derivative of measurement z in respect to intrinic parameters
   */
  void compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta){

    std::lock_guard<std::mutex> lck(mtx);
    // Get our camera parameters
    Eigen::MatrixXd cam_d = camera_values;

    // Calculate distorted coordinates for radial
    double r = std::sqrt(uv_norm(0) * uv_norm(0) + uv_norm(1) * uv_norm(1));
    double r_2 = r * r;
    double r_4 = r_2 * r_2;

    // Jacobian of distorted pixel to normalized pixel
    H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
    double x = uv_norm(0);
    double y = uv_norm(1);
    double x_2 = uv_norm(0) * uv_norm(0);
    double y_2 = uv_norm(1) * uv_norm(1);
    double x_y = uv_norm(0) * uv_norm(1);
    H_dz_dzn(0, 0) = cam_d(0) * ((1 + cam_d(4) * r_2 + cam_d(5) * r_4) + (2 * cam_d(4) * x_2 + 4 * cam_d(5) * x_2 * r_2) +
                                 2 * cam_d(6) * y + (2 * cam_d(7) * x + 4 * cam_d(7) * x));
    H_dz_dzn(0, 1) = cam_d(0) * (2 * cam_d(4) * x_y + 4 * cam_d(5) * x_y * r_2 + 2 * cam_d(6) * x + 2 * cam_d(7) * y);
    H_dz_dzn(1, 0) = cam_d(1) * (2 * cam_d(4) * x_y + 4 * cam_d(5) * x_y * r_2 + 2 * cam_d(6) * x + 2 * cam_d(7) * y);
    H_dz_dzn(1, 1) = cam_d(1) * ((1 + cam_d(4) * r_2 + cam_d(5) * r_4) + (2 * cam_d(4) * y_2 + 4 * cam_d(5) * y_2 * r_2) +
                                 2 * cam_d(7) * x + (2 * cam_d(6) * y + 4 * cam_d(6) * y));

    // Calculate distorted coordinates for radtan
    double x1 = uv_norm(0) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + 2 * cam_d(6) * uv_norm(0) * uv_norm(1) +
                cam_d(7) * (r_2 + 2 * uv_norm(0) * uv_norm(0));
    double y1 = uv_norm(1) * (1 + cam_d(4) * r_2 + cam_d(5) * r_4) + cam_d(6) * (r_2 + 2 * uv_norm(1) * uv_norm(1)) +
                2 * cam_d(7) * uv_norm(0) * uv_norm(1);

    // Compute the Jacobian in respect to the intrinsics
    H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
    H_dz_dzeta(0, 0) = x1;
    H_dz_dzeta(0, 2) = 1;
    H_dz_dzeta(0, 4) = cam_d(0) * uv_norm(0) * r_2;
    H_dz_dzeta(0, 5) = cam_d(0) * uv_norm(0) * r_4;
    H_dz_dzeta(0, 6) = 2 * cam_d(0) * uv_norm(0) * uv_norm(1);
    H_dz_dzeta(0, 7) = cam_d(0) * (r_2 + 2 * uv_norm(0) * uv_norm(0));
    H_dz_dzeta(1, 1) = y1;
    H_dz_dzeta(1, 3) = 1;
    H_dz_dzeta(1, 4) = cam_d(1) * uv_norm(1) * r_2;
    H_dz_dzeta(1, 5) = cam_d(1) * uv_norm(1) * r_4;
    H_dz_dzeta(1, 6) = cam_d(1) * (r_2 + 2 * uv_norm(1) * uv_norm(1));
    H_dz_dzeta(1, 7) = 2 * cam_d(1) * uv_norm(0) * uv_norm(1);
  }

  

  /// Gets the complete intrinsic vector
  Eigen::MatrixXd get_value() { std::lock_guard<std::mutex> lck(mtx);return camera_values; }

  /// Gets the camera matrix
  cv::Matx33d get_K() { std::lock_guard<std::mutex> lck(mtx);return camera_k_OPENCV; }

  /// Gets the camera distortion
  cv::Vec4d get_D() { std::lock_guard<std::mutex> lck(mtx);return camera_d_OPENCV; }

  /// Gets the width of the camera images
  int w() { std::lock_guard<std::mutex> lck(mtx);return _width; }

  /// Gets the height of the camera images
  int h() { std::lock_guard<std::mutex> lck(mtx);return _height; }

  std::shared_ptr<CamModel> clone(){
    auto Clone = std::shared_ptr<CamModel>(new CamModel(w(), h()));
    Clone->set_value(get_value());
    return Clone;
  }

protected:
  // mutex
  mutable std::mutex mtx;

  /// Raw set of camera intrinic values (f_x & f_y & c_x & c_y & k_1 & k_2 & k_3 & k_4)
  Eigen::MatrixXd camera_values;

  /// Camera intrinsics in OpenCV format
  cv::Matx33d camera_k_OPENCV;

  /// Camera distortion in OpenCV format
  cv::Vec4d camera_d_OPENCV;

  /// Width of the camera (raw pixels)
  int _width;

  /// Height of the camera (raw pixels)
  int _height;
};

} // namespace Visual

#endif /* VISUAL_CAMMODEL_H */