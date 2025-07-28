function delta_u = Backstepping_InnerLoop(q_r, q_r_dot, q_r_dot_dot, q, q_dot, q_dot_dot)
% BACKSTEPPING_INNERLOOP
% Inner-loop control law for UAV attitude using backstepping.
%
% This function calculates the control torque increments (delta_u) needed
% to ensure that the actual UAV attitude (yaw, pitch, roll) tracks the
% desired reference attitude. It uses a backstepping control design, where
% errors are defined at two levels (angle error and angular velocity error).
%
% Inputs:
%   q_r         - Reference state vector [8x1]
%   q_r_dot     - Reference angular velocities [8x1]
%   q_r_dot_dot - Reference angular accelerations [8x1]
%   q           - Actual state vector [8x1]
%   q_dot       - Actual angular velocities [8x1]
%   q_dot_dot   - Actual angular accelerations [8x1]
%
% Outputs:
%   delta_u     - Control torque increment for the attitude channels [3x1]

    % --- Backstepping control gains ---
    % High proportional gains (k1) for fast correction of attitude angles
    k1 = diag([360; 360; 360]);
    % Gains (k2) for correction of augmented error (velocity-level)
    k2 = diag([282; 282; 282]);

    % Control effectiveness matrix (inverse of UAV rotational dynamics)
    g_bar = diag([1250 1250 1250]);

    % --- Step 1: Attitude error ---
    % e1 represents the error in yaw, pitch, roll angles
    e1 = q_r(4:6) - q(4:6);

    % --- Step 2: Augmented error ---
    % e2 combines the desired velocity tracking error with a proportional
    % correction on e1. This is typical in backstepping:
    %   e2 = (desired angular velocity + k1*e1) - actual angular velocity
    e2 = q_r_dot(4:6) + k1*e1 - q_dot(4:6);

    % --- Backstepping control law ---
    % The final control law compensates for:
    %   - position error (e1)
    %   - velocity error (k2*e2)
    %   - dynamic effects: actual angular acceleration q_dot_dot
    %   - desired angular acceleration q_r_dot_dot
    %   - additional stabilizing term k1*(q_r_dot - q_dot)
    %
    % Multiplying by inv(g_bar) converts the desired torque increments into
    % commands in terms of control inputs.
    delta_u = inv(g_bar) * ( ...
                    e1 + ...
                    k2*e2 - ...
                    q_dot_dot(4:6) + ...
                    q_r_dot_dot(4:6) + ...
                    k1*(q_r_dot(4:6) - q_dot(4:6)) ...
                );
end