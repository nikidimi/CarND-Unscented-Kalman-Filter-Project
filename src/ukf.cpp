#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  is_initialized_ = false;
 
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  weights_ = VectorXd(2  *n_aug_ + 1);
  
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = 1/(2 * (lambda_ + n_aug_)) ;
  }
  
  P_ <<  1, 0, 0, 0, 0,
         0, 1, 0, 0, 0,
         0, 0, 10, 0, 0,
         0, 0, 0, 10, 0,
         0, 0, 0, 0, 10;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float x = meas_package.raw_measurements_[0] * cos(meas_package.raw_measurements_[1]);
      float y = meas_package.raw_measurements_[0] * sin(meas_package.raw_measurements_[1]);
      x_ << x, y, 1, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }



    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  
  Prediction(dt);
  
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
  else {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;
  
  MatrixXd P_help = P_aug.llt().matrixL();
  
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  double mul = sqrt(lambda_ + n_aug_);
  Xsig_aug.col(0) = x_aug;
  for(int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + mul *  P_help.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - mul *  P_help.col(i);
  }
  
  
  Xsig_pred_ = Xsig_aug.topLeftCorner(n_x_, 2 * n_aug_ + 1);
  
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    double delta_t_sqr = delta_t * delta_t / 2;
    
    if (Xsig_aug(4, i) == 0.0) {
      Xsig_pred_(0, i) += Xsig_aug(2, i) * cos(Xsig_aug(3, i)) * delta_t;
      Xsig_pred_(1, i) += Xsig_aug(2, i) * sin(Xsig_aug(3, i)) * delta_t;
    }
    else {
      double m = Xsig_aug(2, i) / Xsig_aug(4, i);  
      
      Xsig_pred_(0, i) += m * (sin(Xsig_aug(3, i) + delta_t * Xsig_aug(4, i)) - sin(Xsig_aug(3, i)));
      Xsig_pred_(1, i) += m * (-cos(Xsig_aug(3, i) + delta_t * Xsig_aug(4, i)) + cos(Xsig_aug(3, i)));   
    }
    
    Xsig_pred_(0, i) += delta_t_sqr * cos(Xsig_aug(3, i)) * Xsig_aug(5, i);
    Xsig_pred_(1, i) += delta_t_sqr * sin(Xsig_aug(3, i)) * Xsig_aug(5, i);
    Xsig_pred_(2, i) += delta_t * Xsig_aug(5, i);
    Xsig_pred_(3, i) += delta_t * Xsig_aug(4, i) + delta_t_sqr * Xsig_aug(6, i);
    Xsig_pred_(4, i) += delta_t * Xsig_aug(6, i);      
  }
  
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }
  
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    MatrixXd t = Xsig_pred_.col(i) - x_;
    
    while (t(3)> M_PI) t(3)-=2.*M_PI;
    while (t(3)<-M_PI) t(3)+=2.*M_PI;
    
    P_ += weights_(i) * t * t.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;
  
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }
  
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  MatrixXd R = MatrixXd(2, 2);
  R.fill(0.0);
  R(0, 0) = std_laspx_ * std_laspx_;
  R(1, 1) = std_laspy_ * std_laspy_;
  
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    MatrixXd t = Zsig.col(i) - z_pred;
    S += weights_(i) * t * t.transpose();
  }
  
  S += R;
  

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  

  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
    VectorXd z_diff = Zsig.col(i) - z_pred;    
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  
  MatrixXd K = Tc * S.inverse();
  
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  x_ += K * (meas_package.raw_measurements_ - z_pred);
  
  P_ -= K * S * K.transpose(); 
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) { 
  int n_z = 3;
  
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double sq = sqrt(px * px + py * py);
    double psi = Xsig_pred_(3, i);
    double v = Xsig_pred_(2, i);
      
    Zsig(0, i) = sq;
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px * cos(psi) * v + py * sin(psi) * v) /sq;
      
  }
  
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  MatrixXd R = MatrixXd(3, 3);
  R.fill(0.0);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;
  
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    MatrixXd t = Zsig.col(i) - z_pred;
    
    while (t(1)> M_PI) t(1)-=2.*M_PI;
    while (t(1)<-M_PI) t(1)+=2.*M_PI;
    
    S += weights_(i) * t * t.transpose();
  }
  
  S += R;
  

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  

  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
    
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  
  MatrixXd K = Tc * S.inverse();
  
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  x_ += K * (meas_package.raw_measurements_ - z_pred);
  
  P_ -= K * S * K.transpose(); 
}
