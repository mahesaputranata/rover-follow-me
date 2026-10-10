#pragma once
// ================================================================
//  LQRController.hpp
//  Computes optimal wheel velocity commands using LQR.
//
//  Control law:
//    delta_x = x_current - x_ref
//    u       = u_ff - K * delta_x
//
//  where:
//    u       = [v_left, v_right]  in m/s  → sent as SET_VEL
//    delta_x = [dX, dY, dpsi, dv, dr]     → state error
//    K       = 2x5 gain matrix, designed offline (MATLAB)
//    u_ff    = feedforward input, currently constant = v0 for both wheels
//              (no curvature feedforward yet)
//
//  IMPORTANT: K was computed at operating point v0 = 0.2 m/s
//  and psi_0 = 0. It is valid for small deviations around
//  straight forward motion. Re-run the offline design if you
//  change Q, R, or physical parameters significantly.
// ================================================================

#include "RoverState.hpp"
#include <algorithm>
#include <cmath>

// ----------------------------------------------------------------
//  Command output struct
// ----------------------------------------------------------------
struct WheelCommand {
    double v_left;    // left  wheel target velocity (m/s)
    double v_right;   // right wheel target velocity (m/s)
};

// ----------------------------------------------------------------
//  LQRController
// ----------------------------------------------------------------
class LQRController {
public:

    // ----------------------------------------------------------------
    //  K matrix — designed offline (MATLAB), validated in
    //  rover_lqr_framework's rover_sim / followme_sim
    //  Rows: [v_left_cmd, v_right_cmd]
    //  Cols: [X,        Y,         psi,        v,         r      ]
    //  Re-paste here whenever the offline design is re-run
    // ----------------------------------------------------------------
    static constexpr double K[2][5] = {
        {  0.0,  0.0, -0.343673,  1.130740, -0.062116 },  // v_left
        {  0.0,  0.0,  0.343673,  1.130740,  0.062116 },  // v_right
    };

    // ----------------------------------------------------------------
    //  Feedforward input — constant = v0 for both wheels
    //  (no curvature feedforward yet)
    // ----------------------------------------------------------------
    static constexpr double U_FF[2] = { 0.2, 0.2 };

    // ----------------------------------------------------------------
    //  Velocity limits — set to match your rover's safe operating range
    //  Adjust in rover_params.yaml and pass in via RoverConfig later
    // ----------------------------------------------------------------
    static constexpr double V_MAX     =  0.5;   // m/s  max wheel speed
    static constexpr double V_MIN     = -0.5;   // m/s  min wheel speed

    // ----------------------------------------------------------------
    //  compute()
    //  Main control law: u = u_ff - K * delta_x
    //
    //  x_current: state from OdometryNode
    //  x_ref:     reference state from PerceptionBridge
    //  returns:   clamped wheel velocity commands
    // ----------------------------------------------------------------
    WheelCommand compute(const RoverState& x_current,
                         const RoverState& x_ref) const
    {
        // Step 1: compute state error delta_x = x_current - x_ref
        double dx[5];
        dx[0] = x_current.X   - x_ref.X;
        dx[1] = x_current.Y   - x_ref.Y;
        dx[2] = x_current.psi - x_ref.psi;
        dx[3] = x_current.v   - x_ref.v;
        dx[4] = x_current.r   - x_ref.r;

        // Step 2: wrap heading error to [-pi, pi]
        // This prevents the controller from spinning the wrong way
        while (dx[2] >  M_PI) dx[2] -= 2.0 * M_PI;
        while (dx[2] < -M_PI) dx[2] += 2.0 * M_PI;

        // Step 3: u = u_ff - K * delta_x
        // double v_left  = U_FF[0];
        // double v_right = U_FF[1];

	double v_left = x_ref.v;
	double v_right = x_ref.v;

        for (int j = 0; j < 5; ++j) {
            v_left  -= K[0][j] * dx[j];
            v_right -= K[1][j] * dx[j];
        }

        // Step 4: clamp to safe velocity range
        WheelCommand cmd;
        cmd.v_left  = std::clamp(v_left,  V_MIN, V_MAX);
        cmd.v_right = std::clamp(v_right, V_MIN, V_MAX);

        return cmd;
    }
};
