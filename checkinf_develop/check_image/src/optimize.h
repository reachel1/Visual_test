#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <ceres/ceres.h>

#include "data_and_cal.h"

using namespace std;

// NumericDiff
class EpipolarCostFunctor1 {
  public:
    EpipolarCostFunctor1(Eigen::Vector3d x1, Eigen::Vector3d x2, Eigen::Vector3d t_r): observed_x1(x1), observed_x2(x2), t_hat(Visual::skew_x(t_r)){}
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool operator()(const double* const x, double* residuals) const {
      Eigen::Map<const Eigen::Matrix<double,3,1>> T_so3(x);
      // printf("so3:%.2f,%.2f,%.2f\n",T_so3(0),T_so3(1),T_so3(2));
      // std::cout<<T_se3.transpose()<<std::endl;
      Eigen::Matrix3d R = Visual::exp_so3(T_so3);
      Eigen::Matrix3d Ess = t_hat * R;  
      Eigen::Vector3d Epline1 = Ess * observed_x1;
      Eigen::Vector3d Epline2 = Ess.transpose() * observed_x2;
      residuals[0] = sqrt(pow((Epline1.dot(observed_x2) / sqrt(pow(Epline1(0),2) + pow(Epline1(1),2))),2) 
                        + pow((Epline2.dot(observed_x1) / sqrt(pow(Epline2(0),2) + pow(Epline2(1),2))),2));
      
      if(std::isnan(residuals[0])){
          printf("x1: %.3f,%.3f,%.3f, x2: %.3f,%.3f,%.3f\n",observed_x1(0),observed_x1(1),observed_x1(2),observed_x2(0),observed_x2(1),observed_x2(2));
          printf("res now : %.13f\n",sqrt(pow((Epline1.dot(observed_x2) / sqrt(pow(Epline1(0),2) + pow(Epline1(1),2))),2) 
                         + pow((Epline2.dot(observed_x1) / sqrt(pow(Epline2(0),2) + pow(Epline2(1),2))),2)));
      }
      // residuals[0] = k_ - x[0] * y[0] + x[1] * y[1];
      return true;
    }

  private:
    const Eigen::Vector3d observed_x1;
    const Eigen::Vector3d observed_x2;
    const Eigen::Matrix3d t_hat;
};

class EpipolarCostFunctor2 {
  public:
    EpipolarCostFunctor2(Eigen::Vector3d x1, Eigen::Vector3d x2, Eigen::Vector3d t_r): observed_x1(x1), observed_x2(x2), t_hat(Visual::skew_x(t_r)){}
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool operator()(const double* const x, const double* const y, double* residuals) const {
      
      Eigen::Map<const Eigen::Matrix<double,3,1>> T_so31(x);
      Eigen::Map<const Eigen::Matrix<double,3,1>> T_so32(y);
      Eigen::Matrix3d R = Visual::exp_so3(T_so31) * Visual::exp_so3(T_so32);
      Eigen::Matrix3d Ess = t_hat * R;  
      Eigen::Vector3d Epline1 = Ess * observed_x1;
      Eigen::Vector3d Epline2 = Ess.transpose() * observed_x2;
      residuals[0] = sqrt(pow((Epline1.dot(observed_x2) / sqrt(pow(Epline1(0),2) + pow(Epline1(1),2))),2) 
                        + pow((Epline2.dot(observed_x1) / sqrt(pow(Epline2(0),2) + pow(Epline2(1),2))),2));
      
      if(std::isnan(residuals[0])){
          printf("x1: %.3f,%.3f,%.3f, x2: %.3f,%.3f,%.3f\n",observed_x1(0),observed_x1(1),observed_x1(2),observed_x2(0),observed_x2(1),observed_x2(2));
          printf("res now : %.13f\n",sqrt(pow((Epline1.dot(observed_x2) / sqrt(pow(Epline1(0),2) + pow(Epline1(1),2))),2) 
                         + pow((Epline2.dot(observed_x1) / sqrt(pow(Epline2(0),2) + pow(Epline2(1),2))),2)));
      }
      // residuals[0] = k_ - x[0] * y[0] + x[1] * y[1];
      return true;
    }

  private:
    const Eigen::Vector3d observed_x1;
    const Eigen::Vector3d observed_x2;
    const Eigen::Matrix3d t_hat;
};

class EpipolarCostFunctor3 {
  public:
    EpipolarCostFunctor3(Eigen::Vector3d x1, Eigen::Vector3d x2, Eigen::Vector3d t_r): observed_x1(x1), observed_x2(x2), t_hat(Visual::skew_x(t_r)){}
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool operator()(const double* const x, const double* const y, const double* const z, double* residuals) const {
      Eigen::Map<const Eigen::Matrix<double,3,1>> T_so31(x);
      Eigen::Map<const Eigen::Matrix<double,3,1>> T_so32(y);
      Eigen::Map<const Eigen::Matrix<double,3,1>> T_so33(z);
      Eigen::Matrix3d R = Visual::exp_so3(T_so31) * Visual::exp_so3(T_so32) * Visual::exp_so3(T_so33);
      Eigen::Matrix3d Ess = t_hat * R;  
      Eigen::Vector3d Epline1 = Ess * observed_x1;
      Eigen::Vector3d Epline2 = Ess.transpose() * observed_x2;
      residuals[0] = sqrt(pow((Epline1.dot(observed_x2) / sqrt(pow(Epline1(0),2) + pow(Epline1(1),2))),2) 
                        + pow((Epline2.dot(observed_x1) / sqrt(pow(Epline2(0),2) + pow(Epline2(1),2))),2));
      
      if(std::isnan(residuals[0])){
          printf("x1: %.3f,%.3f,%.3f, x2: %.3f,%.3f,%.3f\n",observed_x1(0),observed_x1(1),observed_x1(2),observed_x2(0),observed_x2(1),observed_x2(2));
          printf("res now : %.13f\n",sqrt(pow((Epline1.dot(observed_x2) / sqrt(pow(Epline1(0),2) + pow(Epline1(1),2))),2) 
                         + pow((Epline2.dot(observed_x1) / sqrt(pow(Epline2(0),2) + pow(Epline2(1),2))),2)));
      }
      // residuals[0] = k_ - x[0] * y[0] + x[1] * y[1];
      return true;
    }

  private:
    const Eigen::Vector3d observed_x1;
    const Eigen::Vector3d observed_x2;
    const Eigen::Matrix3d t_hat;
};

class EpipolarCostFunctor4 {
  public:
    EpipolarCostFunctor4(Eigen::Vector3d x1, Eigen::Vector3d x2, Eigen::Vector3d t_r): observed_x1(x1), observed_x2(x2), t_hat(Visual::skew_x(t_r)){}
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool operator()(const double* const x, const double* const y, const double* const z, const double* const w, double* residuals) const {
      Eigen::Map<const Eigen::Matrix<double,3,1>> T_so31(x);
      Eigen::Map<const Eigen::Matrix<double,3,1>> T_so32(y);
      Eigen::Map<const Eigen::Matrix<double,3,1>> T_so33(z);
      Eigen::Map<const Eigen::Matrix<double,3,1>> T_so34(w);
      Eigen::Matrix3d R = Visual::exp_so3(T_so31) * Visual::exp_so3(T_so32) * Visual::exp_so3(T_so33) * Visual::exp_so3(T_so34);
      Eigen::Matrix3d Ess = t_hat * R;  
      Eigen::Vector3d Epline1 = Ess * observed_x1;
      Eigen::Vector3d Epline2 = Ess.transpose() * observed_x2;
      residuals[0] = sqrt(pow((Epline1.dot(observed_x2) / sqrt(pow(Epline1(0),2) + pow(Epline1(1),2))),2) 
                        + pow((Epline2.dot(observed_x1) / sqrt(pow(Epline2(0),2) + pow(Epline2(1),2))),2));
      
      if(std::isnan(residuals[0])){
          printf("x1: %.3f,%.3f,%.3f, x2: %.3f,%.3f,%.3f\n",observed_x1(0),observed_x1(1),observed_x1(2),observed_x2(0),observed_x2(1),observed_x2(2));
          printf("res now : %.13f\n",sqrt(pow((Epline1.dot(observed_x2) / sqrt(pow(Epline1(0),2) + pow(Epline1(1),2))),2) 
                         + pow((Epline2.dot(observed_x1) / sqrt(pow(Epline2(0),2) + pow(Epline2(1),2))),2)));
      }
      // residuals[0] = k_ - x[0] * y[0] + x[1] * y[1];
      return true;
    }

  private:
    const Eigen::Vector3d observed_x1;
    const Eigen::Vector3d observed_x2;
    const Eigen::Matrix3d t_hat;
};


struct EpipolarError
{
	  const Eigen::Vector3d observed_x1;
    const Eigen::Vector3d observed_x2;
    const Eigen::Matrix3d t_hat;

	  EpipolarError(const Eigen::Vector3d x1, const Eigen::Vector3d x2, const Eigen::Vector3d t_r): 
            observed_x1(x1), observed_x2(x2), t_hat(Visual::skew_x(t_r)){}

    template<typename T>
    bool operator()(const T* const x, T* residuals)const
    {
      Eigen::Matrix<T, 3, 1> T_so3(x[0],x[1],x[2]);
      Eigen::Matrix<T, 3, 3> skew_so3;
      skew_so3 << T(0), -x[2], x[1], x[2], T(0), -x[0], -x[1], x[0], T(0);

      T theta = skew_so3.norm();
      // Handle small angle values
      T A, B;
      if (theta < T(1e-7)) {
          A = T(1);
          B = T(0.5);
      } else {
          A = sin(theta) / theta;
          B = (T(1) - cos(theta)) / (theta * theta);
      }
      // compute so(3) rotation
      Eigen::Matrix<T, 3, 3> R;
      Eigen::Matrix<T, 3, 3> eye33 = Eigen::Matrix3d::Identity().cast<T>();
      if (theta == T(0)) {
          R = eye33;
      } else {
          R = eye33 + A * skew_so3 + B * skew_so3 * skew_so3;
      }

      Eigen::Matrix<T, 3, 3> Ess = t_hat.cast<T>() * R;  
      Eigen::Matrix<T, 3, 1> Epline1 = Ess * observed_x1;
      Eigen::Matrix<T, 3, 1> Epline2 = Ess.transpose() * observed_x2;
      
      residuals[0] = sqrt(pow((Epline1.dot(observed_x2.cast<T>()) / sqrt(pow(Epline1(0),2) + pow(Epline1(1),2))),2) 
                        + pow((Epline2.dot(observed_x1.cast<T>()) / sqrt(pow(Epline2(0),2) + pow(Epline2(1),2))),2));

      return true;	// important
    }
};


