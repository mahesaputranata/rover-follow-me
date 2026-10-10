// ================================================================
//  lqr_node.cpp  —  Fix 3 complete
//  Perception bridge converts /bbox_error → x_ref
//  LQR computes u = -K(x_current - x_ref) → /cmd_vel
//
//  Subscribes:  /rover/state  (nav_msgs/Odometry)
//               /bbox_error   (std_msgs/Float32MultiArray)
//                             [offset_x, offset_y, area_ratio, target_valid]
//  Publishes:   /cmd_vel      (geometry_msgs/Twist)
// ================================================================

#include "rover_control/RoverConfig.hpp"
#include "rover_control/RoverState.hpp"
#include "rover_control/LQRController.hpp"

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <memory>
#include <cmath>
#include <algorithm>

// ----------------------------------------------------------------
//  Perception bridge parameters — tune these after testing
//  All in rover_params.yaml eventually, hardcoded here for now
// ----------------------------------------------------------------
static constexpr double BBOX_TARGET_AREA  = 1.0;  // target bbox/frame ratio
static constexpr double BBOX_AREA_TOL     = 0.30;  // dead zone around target
static constexpr double HEADING_GAIN      = 0.4;   // offset_x → psi correction (rad), changed from 0.8 to 0.4
static constexpr double V_FOLLOW          = 0.2;   // forward velocity when following
static constexpr double V_BACKOFF         = -0.15; // reverse velocity when too close
static constexpr int STALE_TICK_LIMIT = 3;
static constexpr double OFFSET_X_DEADBAND = 0.10; // ignoring jitter within +-10% of frame

// new idea -> (experiment): make the response more faster
static constexpr double TURN_ASSIST_OFFSET = 0.15;  // |offset_x| above which hold-zone turn pulses start
static constexpr double PULSE_RATE     = 1.3;   // rad/s during a pulse (wheels +-0.13 m/s, above the 0.08 ESP32 deadband)
static constexpr double PULSE_GAIN     = 0.6;   // seconds of pulse per unit offset
static constexpr double PULSE_MIN_S    = 0.10;
static constexpr double PULSE_MAX_S    = 0.25;
static constexpr double PULSE_SETTLE_S = 0.6;   // wait after a pulse (vision lag) before the next one
static constexpr double PULSE_COAST_S = 0.30;   // after a pulse, command zero so LQR yaw-rate damping cannot counter-steer

class LqrRosNode : public rclcpp::Node
{
public:
    explicit LqrRosNode(const RoverConfig& cfg)
    : Node("lqr_node"), cfg_(cfg), lqr_()
    {
        state_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/rover/state",
            rclcpp::SensorDataQoS(),
            [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
                this->state_callback(msg);
            }
        );

        bbox_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/bbox_error",
            10,
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
                this->bbox_callback(msg);
            }
        );

        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10
        );

        watchdog_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            [this]() { this->watchdog_callback(); }
        );

        RCLCPP_INFO(this->get_logger(),
            "lqr_node ready — perception bridge active");
    }

private:
    // ----------------------------------------------------------------
    //  state_callback — store x_current from odometry
    // ----------------------------------------------------------------
    void state_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        x_current_.X = msg->pose.pose.position.x;
        x_current_.Y = msg->pose.pose.position.y;
        x_current_.v = msg->twist.twist.linear.x;
        x_current_.r = msg->twist.twist.angular.z;

        const auto& q = msg->pose.pose.orientation;
        x_current_.psi = std::atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        );

        state_received_  = true;
        last_state_time_ = this->now();
        compute_and_publish();
    }

    // ----------------------------------------------------------------
    //  bbox_callback — perception bridge
    //  Converts raw bbox error into x_ref for LQR
    //
    //  offset_x   (-1 to +1): person left/right of center
    //    → corrects x_ref.psi (heading reference)
    //    → positive offset_x means person is right → turn right
    //    → x_ref.psi = x_current.psi - offset_x * HEADING_GAIN
    //
    //  area_ratio (0 to 1): proxy for distance
    //    → too small  = person far away  → x_ref.v = V_FOLLOW
    //    → just right = good distance    → x_ref.v = 0 (hold)
    //    → too large  = person too close → x_ref.v = V_BACKOFF
    // ----------------------------------------------------------------
    void bbox_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 4) return;

        const double offset_x     = msg->data[0];
        const double area_ratio   = msg->data[2];
        const double target_valid = msg->data[3];
        bbox_offset_x_ = offset_x;

        double effective_offset_x = offset_x;
        if (std::abs(effective_offset_x) < OFFSET_X_DEADBAND) {
            effective_offset_x = 0.0;
        }

        static constexpr double MAX_PSI_STEP = 0.05;

        if (target_valid < 0.5) {
            x_ref_.psi = x_current_.psi;
            x_ref_.v   = 0.0;
            have_last_psi_ = false;
            speed_zone_ = 0;
            pulse_active_ = false;
        } else {
            double desired_psi =
                x_current_.psi - effective_offset_x * HEADING_GAIN;

            if (!have_last_psi_) {
                x_ref_.psi = desired_psi;
                have_last_psi_ = true;
            } else {
                double delta = std::clamp(
                    desired_psi - last_x_ref_psi_,
                    -MAX_PSI_STEP,
                    MAX_PSI_STEP
                );
                x_ref_.psi = last_x_ref_psi_ + delta;
            }

            last_x_ref_psi_ = x_ref_.psi;

            // Hysteresis: enter a zone at the outer threshold, leave it at the inner one
            static constexpr double FOLLOW_ENTER  = 0.70;
            static constexpr double FOLLOW_EXIT   = 0.80;
            static constexpr double REVERSE_ENTER = 1.40;
            static constexpr double REVERSE_EXIT  = 1.25;

            if (speed_zone_ == 1) {
                if (area_ratio > FOLLOW_EXIT) speed_zone_ = 0;
            } else if (speed_zone_ == -1) {
                if (area_ratio < REVERSE_EXIT) speed_zone_ = 0;
            }
            if (speed_zone_ == 0) {
                if (area_ratio < FOLLOW_ENTER)       speed_zone_ = 1;
                else if (area_ratio > REVERSE_ENTER) speed_zone_ = -1;
            }

            x_ref_.v = (speed_zone_ == 1)  ? V_FOLLOW
                     : (speed_zone_ == -1) ? V_BACKOFF
                     : 0.0;

            // Hold-zone turn assist: setpoints under the ESP32 deadband do nothing, so fire a short strong pulse.
            const double now_s = this->now().seconds();
            if (speed_zone_ == 0 && std::abs(offset_x) > TURN_ASSIST_OFFSET &&
                !pulse_active_ && now_s >= pulse_block_until_s_) {
                const double t_on = std::clamp(PULSE_GAIN * std::abs(offset_x), PULSE_MIN_S, PULSE_MAX_S);
                pulse_dir_ = (offset_x > 0.0) ? -1.0 : 1.0;   // person on the right -> turn right (negative z)
                pulse_end_s_ = now_s + t_on;
                pulse_block_until_s_ = pulse_end_s_ + PULSE_SETTLE_S;
                pulse_active_ = true;
                RCLCPP_INFO(this->get_logger(), "turn pulse dir=%.0f t_on=%.2fs offset=%.2f",
                            pulse_dir_, t_on, offset_x);
            }            
        }

        x_ref_.X = x_current_.X;
        x_ref_.Y = x_current_.Y;
        x_ref_.r = 0.0;
        bbox_received_ = true;
        last_bbox_time_ = this->now();
    }

    // ----------------------------------------------------------------
    //  compute_and_publish — LQR control law
    // ----------------------------------------------------------------
    void compute_and_publish()
    {
        if (!state_received_ || !bbox_received_) {
            return;
        }

        // LQR: u = -K(x_current - x_ref)
        WheelCommand cmd = lqr_.compute(x_current_, x_ref_);

        // Limit wheel velocity command changes.
        if (!have_last_v_) {
            last_v_left_  = cmd.v_left;
            last_v_right_ = cmd.v_right;
            have_last_v_  = true;
        } else {
            const double dl = std::clamp(
                cmd.v_left - last_v_left_,
                -MAX_WHEEL_STEP,
                MAX_WHEEL_STEP);

            const double dr = std::clamp(
                cmd.v_right - last_v_right_,
                -MAX_WHEEL_STEP,
                MAX_WHEEL_STEP);

            cmd.v_left  = last_v_left_  + dl;
            cmd.v_right = last_v_right_ + dr;

            last_v_left_  = cmd.v_left;
            last_v_right_ = cmd.v_right;
        }

        // Convert wheel velocities -> Twist
        const double W = cfg_.rover.track_width_m;

        if (W <= 0.0) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Invalid track_width_m: %.3f",
                W);
            return;
        }

        geometry_msgs::msg::Twist twist;
        twist.linear.x  = (cmd.v_right + cmd.v_left) / 2.0;
        twist.angular.z = (cmd.v_right - cmd.v_left) / W;

        // Hold-zone turn assist pulse (started in bbox_callback)
        if (pulse_active_) {
            const double now_s = this->now().seconds();
            const bool fresh = (this->now() - last_bbox_time_).seconds() < 0.6;
            if (now_s >= pulse_end_s_ || !fresh) {
                pulse_active_ = false;
                pulse_coast_until_s_ = now_s + PULSE_COAST_S;
            } else {
                twist.linear.x  = 0.0;
                twist.angular.z = pulse_dir_ * PULSE_RATE;
            }
        }
        if (!pulse_active_ && this->now().seconds() < pulse_coast_until_s_) {
            twist.linear.x  = 0.0;     // coast: don't let r-damping counter-steer the pulse
            twist.angular.z = 0.0;
        }

        cmd_pub_->publish(twist);

        RCLCPP_DEBUG(
            this->get_logger(),
            "LQR: v_left=%.3f v_right=%.3f -> lin=%.3f ang=%.3f",
            cmd.v_left,
            cmd.v_right,
            twist.linear.x,
            twist.angular.z);
    }

    // ----------------------------------------------------------------
    //  watchdog — stop if state goes silent
    // ----------------------------------------------------------------
    // void watchdog_callback()
    // {
    //     if (!state_received_) return;
    //     auto age = (this->now() - last_state_time_).seconds();
    //     if (age > 0.5) {
    //         RCLCPP_WARN(this->get_logger(),
    //             "No state update for %.2fs — publishing STOP", age);
    //         cmd_pub_->publish(geometry_msgs::msg::Twist());
    //     }
    // }
    void watchdog_callback()
    {
        bool state_stale_now = state_received_ && (this->now() - last_state_time_).seconds() > 0.5;
        bool bbox_stale_now = bbox_received_ && (this->now() - last_bbox_time_).seconds() > 0.5;

        state_stale_count_ = state_stale_now ? state_stale_count_ + 1 : 0;
        bbox_stale_count_ = bbox_stale_now ? bbox_stale_count_ + 1 : 0;

        if(state_stale_count_ >= STALE_TICK_LIMIT || bbox_stale_count_ >= STALE_TICK_LIMIT){
            if(!stopped_due_to_watchdog_){
                RCLCPP_WARN(this->get_logger(),
                    "Watchdog: sustained staleness (state=%d bbox=%d ticks) — STOP",
                    state_stale_count_, bbox_stale_count_);
                cmd_pub_ -> publish(geometry_msgs::msg::Twist());
                stopped_due_to_watchdog_ = true;
            }
        if(state_stale_count_ >= STALE_TICK_LIMIT) state_received_ = false;
        if(bbox_stale_count_ >= STALE_TICK_LIMIT) bbox_received_ = false;
        
            
        } else{
            stopped_due_to_watchdog_ = false;
        }
    }

    RoverConfig    cfg_;
    LQRController  lqr_;

    RoverState     x_current_;
    RoverState     x_ref_;
    bool           state_received_ = false;
    bool           bbox_received_  = false;
    float bbox_offset_x_ = 0.0f; 
    float bbox_offset_y_ = 0.0f; // dead
    float bbox_area_ratio_ = 0.0f; // dead
    int  state_stale_count_      = 0;
    int  bbox_stale_count_       = 0;
    bool stopped_due_to_watchdog_ = false;
    double last_x_ref_psi_ = 0.0;
    bool have_last_psi_ = false;
    int speed_zone_ = 0; // 0 = hold, +1 = follow, -1 = reverse (hysteresis state)

    bool   pulse_active_ = false; // (experiment) make response more faster
    double pulse_dir_ = 0.0;
    double pulse_end_s_ = 0.0;
    double pulse_block_until_s_ = 0.0;
    double pulse_coast_until_s_ = 0.0;

    double last_v_left_ = 0.0; 
    double last_v_right_ = 0.0;
    bool have_last_v_ = false;
    static constexpr double MAX_WHEEL_STEP = 0.03;
    rclcpp::Time   last_state_time_;
    rclcpp::Time last_bbox_time_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr           state_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr  bbox_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr            cmd_pub_;
    rclcpp::TimerBase::SharedPtr                                       watchdog_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    std::string config_path =
        ament_index_cpp::get_package_share_directory("rover_control")
        + "/config/rover_params.yaml";
    if (argc > 1) config_path = argv[1];

    RoverConfig cfg;
    try {
        cfg = RoverConfig::load(config_path);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("lqr_node"),
            "Failed to load config: %s", e.what());
        return 1;
    }

    rclcpp::spin(std::make_shared<LqrRosNode>(cfg));
    rclcpp::shutdown();
    return 0;
}
