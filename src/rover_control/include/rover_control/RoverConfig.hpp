#pragma once
// ================================================================
//  RoverConfig.hpp
//  Loads rover_params.yaml and exposes typed structs.
//  Depends on: yaml-cpp  (sudo apt install libyaml-cpp-dev)
// ================================================================

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

// ----------------------------------------------------------------
//  Parameter structs — one per yaml section
// ----------------------------------------------------------------
struct RoverParams {
    double track_width_m;
    double wheel_diameter_m;
    int    ticks_per_rev;
};

struct OdometryParams {
    double publish_rate_hz;
    double initial_x;
    double initial_y;
    double initial_psi;
};

struct LQRParams {
    double              v0;
    std::vector<double> Q_diag;   // length 5: [X, Y, psi, v, r]
    std::vector<double> R_diag;   // length 2: [v_left, v_right]
};

struct SerialParams {
    std::string port;
    int         baud_rate;
};

// ----------------------------------------------------------------
//  Top-level config container
// ----------------------------------------------------------------
struct RoverConfig {
    RoverParams    rover;
    OdometryParams odometry;
    LQRParams      lqr;
    SerialParams   serial;

    // Derived constants — computed once after loading
    double meters_per_tick;   // = (pi * wheel_diameter_m) / ticks_per_rev
    double wheel_radius_m;    // = wheel_diameter_m / 2.0

    // Factory: load from yaml file path
    static RoverConfig load(const std::string& path);
};

// ----------------------------------------------------------------
//  Implementation
// ----------------------------------------------------------------
inline RoverConfig RoverConfig::load(const std::string& path)
{
    YAML::Node root = YAML::LoadFile(path);
    RoverConfig cfg;

    // --- rover section ---
    auto r = root["rover"];
    cfg.rover.track_width_m    = r["track_width_m"].as<double>();
    cfg.rover.wheel_diameter_m = r["wheel_diameter_m"].as<double>();
    cfg.rover.ticks_per_rev    = r["ticks_per_rev"].as<int>();

    // --- odometry section ---
    auto o = root["odometry"];
    cfg.odometry.publish_rate_hz = o["publish_rate_hz"].as<double>();
    cfg.odometry.initial_x       = o["initial_x"].as<double>();
    cfg.odometry.initial_y       = o["initial_y"].as<double>();
    cfg.odometry.initial_psi     = o["initial_psi"].as<double>();

    // --- lqr section ---
    auto l = root["lqr"];
    cfg.lqr.v0     = l["v0"].as<double>();
    cfg.lqr.Q_diag = l["Q_diag"].as<std::vector<double>>();
    cfg.lqr.R_diag = l["R_diag"].as<std::vector<double>>();

    // --- serial section ---
    auto s = root["serial"];
    cfg.serial.port      = s["port"].as<std::string>();
    cfg.serial.baud_rate = s["baud_rate"].as<int>();

    // --- derived constants ---
    constexpr double PI = 3.14159265358979323846;
    cfg.meters_per_tick = (PI * cfg.rover.wheel_diameter_m)
                          / static_cast<double>(cfg.rover.ticks_per_rev);
    cfg.wheel_radius_m  = cfg.rover.wheel_diameter_m / 2.0;

    return cfg;
}
