// ================================================================
//  odom_node.cpp
//  ROS2 C++ node — odometry integration.
//
//  Subscribes:  /esp32/odom  (std_msgs/String)
//               Format: "ODOM <dA_ticks> <dB_ticks> <dt_ms>"
//               Published by bridge_node.py serial_reader
//
//  Publishes:   /rover/state  (nav_msgs/Odometry)
//               Full state: X, Y, psi, v, r
//               Consumed by lqr_node
//
//  Config:      rover_params.yaml loaded at startup
// ================================================================

#include "rover_control/RoverConfig.hpp"
#include "rover_control/OdometryNode.hpp"

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <std_msgs/msg/string.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/quaternion.hpp>

#include <cmath>
#include <string>
#include <memory>

// ----------------------------------------------------------------
//  Euler yaw → quaternion (for nav_msgs/Odometry)
// ----------------------------------------------------------------
geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw)
{
    geometry_msgs::msg::Quaternion q;
    q.x = 0.0;
    q.y = 0.0;
    q.z = std::sin(yaw / 2.0);
    q.w = std::cos(yaw / 2.0);
    return q;
}

// ----------------------------------------------------------------
//  OdomRosNode
// ----------------------------------------------------------------
class OdomRosNode : public rclcpp::Node
{
public:
    explicit OdomRosNode(const RoverConfig& cfg)
    : Node("odom_node"), odom_(cfg)
    {
        // Subscribe to raw ODOM strings from bridge_node.py
        sub_ = this->create_subscription<std_msgs::msg::String>(
            "/esp32/odom",
            rclcpp::SensorDataQoS(),
            [this](const std_msgs::msg::String::SharedPtr msg) {
                this->odom_callback(msg->data);
            }
        );

        // Publish full state as nav_msgs/Odometry
        pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/rover/state", 10
        );

        RCLCPP_INFO(this->get_logger(),
            "odom_node ready — listening on /esp32/odom");
    }

private:
    // ----------------------------------------------------------------
    //  odom_callback
    //  Called every time bridge_node.py forwards an ODOM line.
    //  Integrates state and publishes /rover/state.
    // ----------------------------------------------------------------
    void odom_callback(const std::string& line)
    {
        bool updated = odom_.parse_and_update(line);
        if (!updated) return;

        const RoverState& s = odom_.state();

        // Build nav_msgs/Odometry message
        nav_msgs::msg::Odometry msg;
        msg.header.stamp    = this->now();
        msg.header.frame_id = "odom";
        msg.child_frame_id  = "base_link";

        // Position
        msg.pose.pose.position.x = s.X;
        msg.pose.pose.position.y = s.Y;
        msg.pose.pose.position.z = 0.0;

        // Orientation (yaw only — 2D rover)
        msg.pose.pose.orientation = yaw_to_quaternion(s.psi);

        // Velocity in body frame
        msg.twist.twist.linear.x  = s.v;
        msg.twist.twist.angular.z = s.r;

        pub_->publish(msg);

        RCLCPP_DEBUG(this->get_logger(),
            "state: X=%.3f Y=%.3f psi=%.3f v=%.3f r=%.3f",
            s.X, s.Y, s.psi, s.v, s.r);
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr  pub_;
    OdometryNode odom_;
};

// ----------------------------------------------------------------
//  main
// ----------------------------------------------------------------
int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    // Load config — find yaml via ROS2 share directory
    // Override with CLI argument if provided
    std::string config_path =
        ament_index_cpp::get_package_share_directory("rover_control")
        + "/config/rover_params.yaml";

    if (argc > 1) config_path = argv[1];

    RoverConfig cfg;
    try {
        cfg = RoverConfig::load(config_path);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("odom_node"),
            "Failed to load config: %s", e.what());
        return 1;
    }

    RCLCPP_INFO(rclcpp::get_logger("odom_node"),
        "Config loaded — track_width=%.3f m  meters_per_tick=%.6f",
        cfg.rover.track_width_m, cfg.meters_per_tick);

    rclcpp::spin(std::make_shared<OdomRosNode>(cfg));
    rclcpp::shutdown();
    return 0;
}
