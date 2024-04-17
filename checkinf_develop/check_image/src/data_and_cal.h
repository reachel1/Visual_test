#ifndef VISUAL_DATACAL_H
#define VISUAL_DATACAL_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <vector>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

namespace Visual {

    struct CamData {

        /// Timestamp of the reading
        double timestamp;

        /// Raw image we have collected for each camera
        cv::Mat image;

        /// Sort function to allow for using of STL containers
        bool operator<(const CamData &other) const {
            return timestamp < other.timestamp;
        }
    };


    struct Feature{

        /// Unique ID of this feature
        size_t featid;

        /// If this feature should be deleted
        bool to_delete;

        /// UV coordinates that this feature has been seen from
        std::vector<Eigen::Vector2f> uvs;

        /// UV normalized coordinates that this feature has been seen from
        std::vector<Eigen::Vector2f> uvs_norm;

        /// Timestamps of each UV measurement
        std::vector<double> timestamps;

    };

    inline Eigen::Matrix<double, 3, 3> skew_x(const Eigen::Matrix<double, 3, 1> &w) {
        Eigen::Matrix<double, 3, 3> w_x;
        w_x << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
        return w_x;
    }

    inline Eigen::Matrix<double, 3, 1> log_so3(const Eigen::Matrix<double, 3, 3> &R) {

        // note switch to base 1
        double R11 = R(0, 0), R12 = R(0, 1), R13 = R(0, 2);
        double R21 = R(1, 0), R22 = R(1, 1), R23 = R(1, 2);
        double R31 = R(2, 0), R32 = R(2, 1), R33 = R(2, 2);

        // Get trace(R)
        const double tr = R.trace();
        Eigen::Vector3d omega;

        // when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
        // we do something special
        if (tr + 1.0 < 1e-10) {
            if (std::abs(R33 + 1.0) > 1e-5)
            omega = (M_PI / sqrt(2.0 + 2.0 * R33)) * Eigen::Vector3d(R13, R23, 1.0 + R33);
            else if (std::abs(R22 + 1.0) > 1e-5)
            omega = (M_PI / sqrt(2.0 + 2.0 * R22)) * Eigen::Vector3d(R12, 1.0 + R22, R32);
            else
            // if(std::abs(R.r1_.x()+1.0) > 1e-5)  This is implicit
            omega = (M_PI / sqrt(2.0 + 2.0 * R11)) * Eigen::Vector3d(1.0 + R11, R21, R31);
        } else {
            double magnitude;
            const double tr_3 = tr - 3.0; // always negative
            if (tr_3 < -1e-7) {
            double theta = acos((tr - 1.0) / 2.0);
            magnitude = theta / (2.0 * sin(theta));
            } else {
            // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            // see https://github.com/borglab/gtsam/issues/746 for details
            magnitude = 0.5 - tr_3 / 12.0;
            }
            omega = magnitude * Eigen::Vector3d(R32 - R23, R13 - R31, R21 - R12);
        }

        return omega;
    }

    inline Eigen::Matrix<double, 3, 3> exp_so3(const Eigen::Matrix<double, 3, 1> &w) {
        // get theta
        Eigen::Matrix<double, 3, 3> w_x = skew_x(w);
        double theta = w.norm();
        // Handle small angle values
        double A, B;
        if (theta < 1e-7) {
            A = 1;
            B = 0.5;
        } else {
            A = sin(theta) / theta;
            B = (1 - cos(theta)) / (theta * theta);
        }
        // compute so(3) rotation
        Eigen::Matrix<double, 3, 3> R;
        if (theta == 0) {
            R = Eigen::MatrixXd::Identity(3, 3);
        } else {
            R = Eigen::MatrixXd::Identity(3, 3) + A * w_x + B * w_x * w_x;
        }
        return R;
    }

    inline Eigen::Matrix4d Inv_se3(const Eigen::Matrix4d &T) {
        Eigen::Matrix4d Tinv = Eigen::Matrix4d::Identity();
        Tinv.block(0, 0, 3, 3) = T.block(0, 0, 3, 3).transpose();
        Tinv.block(0, 3, 3, 1) = -Tinv.block(0, 0, 3, 3) * T.block(0, 3, 3, 1);
        return Tinv;
    }
} // namespace Visual

#endif // VISUAL_DATACAL_H