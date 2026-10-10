#pragma once
// ================================================================
//  RoverState.hpp
//  Holds the full state vector x = [X, Y, psi, v, r]
//  This is what the odometry node produces and LQR consumes.
// ================================================================

#include <cmath>

struct RoverState {
    double X;     // global x position (m)
    double Y;     // global y position (m)
    double psi;   // heading angle (rad)
    double v;     // forward velocity in body frame (m/s)
    double r;     // yaw rate (rad/s)

    // Timestamp of this state estimate (seconds)
    double timestamp_s = 0.0;

    // Convenience: wrap psi to [-pi, pi]
    void normalize_psi() {
        while (psi >  M_PI) psi -= 2.0 * M_PI;
        while (psi < -M_PI) psi += 2.0 * M_PI;
    }
};
