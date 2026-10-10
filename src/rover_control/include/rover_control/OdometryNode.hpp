#pragma once
// ================================================================
//  OdometryNode.hpp
//  Parses ESP32 ODOM messages and integrates the state vector.
//
//  ESP32 streams at 50 Hz:
//    "ODOM <dA_ticks> <dB_ticks> <dt_ms>"
//
//  Internally implements the discrete kinematic update derived
//  from the nonlinear model:
//    X_{k+1}   = X_k   + d_c * cos(psi_k + delta_psi/2)
//    Y_{k+1}   = Y_k   + d_c * sin(psi_k + delta_psi/2)
//    psi_{k+1} = psi_k + delta_psi
//    v_k       = d_c / dt
//    r_k       = delta_psi / dt
// ================================================================

#include "RoverConfig.hpp"
#include "RoverState.hpp"
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>

class OdometryNode {
public:
    // ----------------------------------------------------------------
    //  Constructor: inject config, set initial state
    // ----------------------------------------------------------------
    explicit OdometryNode(const RoverConfig& cfg)
        : cfg_(cfg)
    {
        state_.X         = cfg.odometry.initial_x;
        state_.Y         = cfg.odometry.initial_y;
        state_.psi       = cfg.odometry.initial_psi;
        state_.v         = 0.0;
        state_.r         = 0.0;
        state_.timestamp_s = 0.0;
    }

    // ----------------------------------------------------------------
    //  parse_and_update()
    //  Call this for every line received from the ESP32.
    //  Returns true if the line was an ODOM message and state updated.
    // ----------------------------------------------------------------
    bool parse_and_update(const std::string& line)
    {
        // Only handle ODOM messages
        if (line.rfind("ODOM", 0) != 0) return false;

        long   dA = 0, dB = 0;
        double dt_ms = 0.0;

        std::istringstream ss(line);
        std::string tag;
        ss >> tag >> dA >> dB >> dt_ms;

        if (ss.fail()) {
            std::cerr << "[OdometryNode] Bad ODOM line: " << line << "\n";
            return false;
        }

        integrate(dA, dB, dt_ms / 1000.0);   // convert ms -> seconds
        return true;
    }

    // ----------------------------------------------------------------
    //  Current state accessor (read-only)
    // ----------------------------------------------------------------
    const RoverState& state() const { return state_; }

    // ----------------------------------------------------------------
    //  Reset to a known pose (useful for re-localisation)
    // ----------------------------------------------------------------
    void reset(double x = 0.0, double y = 0.0, double psi = 0.0) {
        state_.X   = x;
        state_.Y   = y;
        state_.psi = psi;
        state_.v   = 0.0;
        state_.r   = 0.0;
    }

private:
    // ----------------------------------------------------------------
    //  Core discrete kinematic update
    //  Implements the midpoint integration from the model derivation.
    // ----------------------------------------------------------------
    void integrate(long dA, long dB, double dt_s)
    {
        if (dt_s <= 0.0) return;

        // Step 1: ticks -> wheel arc distances (note: 08/10/26 d_L change so it use dB and d_R changed so it use dA)
        const double d_L = static_cast<double>(dB) * cfg_.meters_per_tick;
        const double d_R = static_cast<double>(dA) * cfg_.meters_per_tick;

        // Step 2: wheel arcs -> rover center motion
        const double d_c       = (d_L + d_R) / 2.0;
        const double delta_psi = (d_R - d_L) / cfg_.rover.track_width_m;

        // Step 3: midpoint heading for more accurate integration
        const double psi_mid = state_.psi + delta_psi / 2.0;

        // Step 4: global position update (discrete kinematics)
        state_.X   += d_c * std::cos(psi_mid);
        state_.Y   += d_c * std::sin(psi_mid);
        state_.psi += delta_psi;

        // Step 5: velocity estimates (for LQR state vector)
        state_.v = d_c / dt_s;
        state_.r = delta_psi / dt_s;

        // Keep psi in [-pi, pi]
        state_.normalize_psi();

        // Advance internal clock
        state_.timestamp_s += dt_s;
    }

    // ----------------------------------------------------------------
    //  Members
    // ----------------------------------------------------------------
    const RoverConfig& cfg_;
    RoverState         state_;
};
