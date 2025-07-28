function delta_u = Backstepping_InnerLoop(q_r, q_r_dot, q_r_dot_dot, q, q_dot, q_dot_dot)
% BACKSTEPPING_INNERLOOP
% This function implements the inner-loop control law for UAV attitude
% using a backstepping-based approach. 
% It calculates the control input increment (delta_u) for roll, pitch, and yaw.

    % --- Control gains for backstepping ---
    % k1: proportional-like gains for attitude tracking
    k1 = diag([258; 258; 258]);  % high gain for fast convergence
    % k2: additional gains for stabilizing the second error term
    k2 = diag([222; 222; 222]);

    % --- Control effectiveness matrix for inner loop ---
    % g_bar represents the approximate dynamic influence of torques on angular accelerations
    g_bar = diag([950 950 950]);

    % --- Compute errors ---
    % e1: attitude error (desired angles - actual angles)
    e1 = q_r(4:6) - q(4:6);

    % e2: derivative error (combines angular velocity tracking and stabilization)
    %     backstepping introduces an augmented error term:
    %     e2 = q_r_dot + k1*e1 - q_dot
    e2 = q_r_dot(4:6) + k1*e1 - q_dot(4:6);

    % --- Backstepping control law ---
    % The control increment delta_u compensates for:
    %   - attitude error (e1)
    %   - derivative error (k2*e2)
    %   - current angular accelerations q_dot_dot
    %   - desired angular accelerations q_r_dot_dot
    %   - and includes a corrective term k1*(q_r_dot - q_dot)
    % g_bar^-1 ensures correct scaling with system dynamics
    delta_u = inv(g_bar) * ( ...
                    e1 + ...
                    k2*e2 - ...
                    q_dot_dot(4:6) + ...
                    q_r_dot_dot(4:6) + ...
                    k1*(q_r_dot(4:6) - q_dot(4:6)) ...
               );
end
