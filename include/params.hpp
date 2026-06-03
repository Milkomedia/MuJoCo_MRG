#ifndef PARAMS_H
#define PARAMS_H

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <chrono>
#include <Eigen/Dense>
#include <mujoco/mujoco.h>

static inline constexpr double inv_sqrt2 = 0.7071067811865474617150084668537601828575;  // 1/sqrt(2)
static inline constexpr double sqrt2 = 1.41421356237309504880168872420969807856967187537694;  // sqrt(2)

namespace param {

// ===== UAV Parameters =====
inline constexpr double J[9] = {0.27, 0.00, 0.00,
                                0.00, 0.49, 0.00,
                                0.00, 0.00, 0.76}; // nominal UAV moment of inertia [kg m^2]

inline constexpr double PAYLOAD_MASS = 0.8; // additional payload mass [kg]
inline constexpr double M  = 4.8949497 + PAYLOAD_MASS;
inline constexpr double G  = 9.80665;

inline constexpr double SATURATION_THRUST = 20.0; // [N]

// ===== SE3 controlelr gains ====
// Control Parameters
inline constexpr double kX[3] = {40.0, 40.0, 65.0}; // Position gain [x, y, z]
inline constexpr double kV[3] = {20.0, 20.0, 80.0}; // Velocity gain [x, y, z]
inline constexpr double kR[3] = {120., 120., 5.5}; // Rotational gain [roll, pitch, yaw]
inline constexpr double kW[3] = {15.0, 15.0,  2.5}; // angular Velocity gain [roll, pitch, yaw]

// Integral Parameters
inline constexpr double kI  = 0.0; // Attitude integral gain for roll & pitch
inline constexpr double kyI = 0.0; // Attitude integral gain for yaw
inline constexpr double kIX = 0.3; // Position integral gain

// ===== Motor model =====
inline constexpr double PWM_A            = 45.0;  // propeller thrust[N] = A * pwm^2 + B
inline constexpr double PWM_B            = 8.0;   // propeller thrust[N] = A * pwm^2 + B
inline constexpr double PWM_ZETA         = 0.02;  // propeller torque[Nm] = zeta * thrust
inline constexpr double rotor_dir[4]     = {1.0, -1.0, 1.0, -1.0}; // propeller torque direction

inline constexpr double nominal_PWM_A    = 45.0;
inline constexpr double nominal_PWM_B    = 8.0;
inline constexpr double nominal_PWM_ZETA = 0.02;

// Sequential Allocation parameters
inline constexpr double SERVO_DELAY_ALPHA = 0.093158;  // yaw trimming
inline constexpr double SERVO_DELAY_BETA  = 1.0 - SERVO_DELAY_ALPHA;
inline constexpr double TAUZ_MIN = -3.0; // saturation ref [Nm]
inline constexpr double TAUZ_MAX =  3.0;

// ===== arm DH parameters =====
inline constexpr double B2BASE_THETA[4] = {-0.25*M_PI, -0.75*M_PI, 0.75*M_PI, 0.25*M_PI};
inline constexpr double B2BASE_ALPHA[4] = {M_PI, M_PI, M_PI, M_PI};
inline constexpr double B2BASE_X[4]     = { 0.12*inv_sqrt2, -0.12*inv_sqrt2, -0.12*inv_sqrt2,  0.12*inv_sqrt2}; // x-distance from the body frame to each base frame [m]
inline constexpr double B2BASE_Y[4]     = {-0.12*inv_sqrt2, -0.12*inv_sqrt2,  0.12*inv_sqrt2,  0.12*inv_sqrt2}; // y-distance from the body frame to each base frame [m]
inline constexpr double B2BASE_A[4]     = {0.120, 0.120, 0.120, 0.120};
inline constexpr double DH_ARM_A[5]     = {0.134, 0.115, 0.110, 0.024, 0.071};
inline constexpr double DH_ARM_ALPHA[5] = {M_PI/2.0, 0.0, 0.0, M_PI/2.0, 0.0};

// ===== initial arm positions =====
inline const     Eigen::Vector2d r1_init       = Eigen::Vector2d(0.219411255, -1.0*M_PI/4.0); // rotor-1 inital position
inline const     Eigen::Vector2d r2_init       = Eigen::Vector2d(0.219411255, -3.0*M_PI/4.0); // rotor-2 inital position
inline const     Eigen::Vector2d r3_init       = Eigen::Vector2d(0.219411255,  3.0*M_PI/4.0); // rotor-3 inital position
inline const     Eigen::Vector2d r4_init       = Eigen::Vector2d(0.219411255,  1.0*M_PI/4.0); // rotor-4 inital position
inline constexpr double r_z_position           = -0.24; // rotor z position [m]

// ===== Workspace constraint for collision check =====
inline constexpr double D_ROTOR      = 0.2;   // propeller diameter [m]
inline constexpr double RHO_MIN      = 0.1506; // Minimum distance arm can extend from the base [m]
inline constexpr double RHO_MAX      = 0.2925; // Maximum distance arm can extend from the base [m]
inline constexpr double ALPHA_MIN[4] = {-105.0 * M_PI/180.0, -195.0 * M_PI/180.0,  75.0 * M_PI/180.0, -15.0 * M_PI/180.0};
inline constexpr double ALPHA_MAX[4] = {  15.0 * M_PI/180.0,  -75.0 * M_PI/180.0, 195.0 * M_PI/180.0, 105.0 * M_PI/180.0};

// ===== CoM estimating parameter =====
inline constexpr double LINK_MASS[5] = {0.187053, 0.06829, 0.0207574, 0.0510015, 0.1867}; // link mass [kg]
inline constexpr double CENTER_MASS  = 3.2397421;   // center body + bong + center mass [kg]
inline constexpr double TOTAL_MASS   = CENTER_MASS + 4.0*(LINK_MASS[0]+LINK_MASS[1]+LINK_MASS[2]+LINK_MASS[3]+LINK_MASS[4]); // strider mass (same as M) [kg]
inline constexpr double LINK_COM_DIST[5] = {-0.040, -0.031, -0.055, -0.012, -0.020};     // link com distance [m]

// ===== MoI estimating parameter =====
inline constexpr double LINK_INERTIAL_POS[5][3] = {
  {-0.0628864, -0.00030071,  0.0115046},
  {-0.0812353,  0.0,         0.00033993},
  {-0.0545139, -0.00013889,  0.00012511},
  {-0.0120002,  0.0005559,  -0.000556},
  { 0.0497922,  0.0,         0.00013454}}; // Inertial pos of each link frame [m]
inline constexpr double LINK_INERTIAL_QUAT[5][4] = {
  {0.432962, 0.567439, 0.525068, 0.46353},
  {0.501818, 0.498175, 0.498175, 0.501818},
  {0.503095, 0.499879, 0.497035, 0.499972},
  {0.270139, 0.652173, 0.271057, 0.654388},
  {1.0,      0.0,      0.000236013, 0.0}}; // Inertial quat of each link frame, wxyz order
inline constexpr double LINK_INERTIA_DIAG[5][3] = {
  {0.00164276, 0.00160619, 0.000141896},
  {0.000183,   0.000171008, 2.96325e-05},
  {4.29713e-05, 3.86412e-05, 7.76758e-06},
  {2.88701e-05, 2.767e-05,   2.05299e-05},
  {0.00364513, 0.0019397,    0.00192912}}; // Principal inertia of each link [kg.m^2]
inline const Eigen::Matrix3d CENTER_INERTIA = (Eigen::Matrix3d() <<
  0.00761191, 0.0,        0.00124125,
  0.0,        0.23172556, 0.0,
  0.00124125, 0.0,        0.23103643).finished(); // BODY + bar + center_mass inertia about body origin; [kg.m^2]

// ===== Frequencies & timesteps =====
inline constexpr double CTRL_HZ   = 400.0;  // flight contol
inline constexpr double SIM_HZ    = 1000.0; // mujoco simul
inline constexpr double VIEWER_HZ = 30.0;   // mujoco viewer

inline const std::chrono::steady_clock::duration VIEWER_DT = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / VIEWER_HZ));
inline constexpr double CTRL_DT   = 1.0 / CTRL_HZ;
inline constexpr double SIM_DT    = 1.0 / SIM_HZ;

// ===== MPC parameters  =====
inline constexpr double      MPC_STEP_DT = 1.0 / 250.0; // This value must be same as >> DT << on params.py
inline constexpr std::size_t N_STEPS_REQ = 10; // This value must be less than >> N << on params.py
inline constexpr std::chrono::steady_clock::duration MPC_TIMEOUT_DURATUION = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(static_cast<double>(N_STEPS_REQ-1) * MPC_STEP_DT));

// ===== actuator delay (sim<->real validation) =====
inline constexpr double ROTOR_DELAY_TAU    = 0.025; // [sec]
inline const     double ROTOR_DELAY_ALPHA  = std::exp(-CTRL_DT / ROTOR_DELAY_TAU);
inline const     double ROTOR_DELAY_BETA   = 1.0 - ROTOR_DELAY_ALPHA;             

inline constexpr double ARM_DELAY_TAU    = 0.03; // [sec]
inline const     double ARM_DELAY_ALPHA  = std::exp(-CTRL_DT / ARM_DELAY_TAU);
inline const     double ARM_DELAY_BETA   = 1.0 - ARM_DELAY_ALPHA;             

inline constexpr double BASE_DELAY_TAU   = 0.05; // [sec]
inline const     double BASE_DELAY_ALPHA = std::exp(-CTRL_DT / BASE_DELAY_TAU);
inline const     double BASE_DELAY_BETA  = 1.0 - BASE_DELAY_ALPHA;             

// ===== state noise input (sim<->real validation) =====
inline constexpr std::uint64_t NOISE_SEED = 42;

inline constexpr double POS_NOISE_SIGMA   = 0.001; // white noise std,  [m]
inline constexpr double POS_BIAS_RW       = 0.0001; // bias random-walk, [m/sqrt(s)]
inline constexpr double VEL_NOISE_SIGMA   = 0.001; // white noise std, [m/s]
inline constexpr double ACC_NOISE_SIGMA   = 0.001; // white noise std, [m/s^2]

inline constexpr double RP_NOISE_SIGMA    = 0.01 * M_PI/180.0;  // white noise std (roll/pitch), [rad]
inline constexpr double YAW_NOISE_SIGMA   = 0.02 * M_PI/180.0;  // white noise std (yaw), [rad]
inline constexpr double RP_NOISE_BIAS_RW  = 0.002 * M_PI/180.0;  // [rad/sqrt(s)]
inline constexpr double YAW_NOISE_BIAS_RW = 0.005 * M_PI/180.0;  // [rad/sqrt(s)]

inline constexpr double OMEGA_NOISE_SIGMA = 2.5 * M_PI/180.0;  // white noise std, [rad/s]

inline constexpr double ARM_NOISE_SIGMA   = 0.10 * M_PI/180.0;  // white noise std,  [rad/s]
inline constexpr double ARM_BIAS_RW       = 0.02 * M_PI/180.0;  // bias random-walk, [rad/s/sqrt(s)]

} // namespace param

#endif // PARAMS_H