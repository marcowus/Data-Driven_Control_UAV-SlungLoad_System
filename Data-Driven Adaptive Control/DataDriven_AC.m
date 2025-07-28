clc
clear 
close all;

%% Parameters and Initialization
Ts = 0.001; % Sampling time (1 ms)
time = 50;  % Total simulation time (s)
ga = 9.81;  % Gravity acceleration (m/s^2)
M = 0.4;    % Mass of UAV (kg)
m = 0.03;   % Mass of the suspended load (kg)

% State vectors: 
% q = [x; y; z; psi; theta; phi; alpha; beta]
% (position, attitude angles, and load swing angles)
q   = zeros(8, time/Ts);
qd  = zeros(8, time/Ts);  % First derivative of q (velocity)
qdd = zeros(8, time/Ts);  % Second derivative of q (acceleration)

swing = zeros(2, time/Ts); % Stores swing angles in degrees for plotting

% Initial conditions (slightly offset from zero)
q(:,1) = [0.2; 0; 0.02; 0; 0; 0; 0.0001; 0];
q(:,2) = [0.2; 0; 0.02; 0; 0; 0; 0.0001; 0];

% Control torques for UAV
tau = zeros(6, time/Ts);
delta_tau = zeros(3, time/Ts); % Incremental control for outer loop

% Reference trajectory (initially same as current state)
q_r = zeros(8, time/Ts);
q_r(:,1) = [0.2; 0; 0.02; 0; 0; 0; 0; 0];
q_r(:,2) = [0.2; 0; 0.02; 0; 0; 0; 0; 0];
qd_r = zeros(8,time/Ts);   % Reference velocities
qdd_r = zeros(8,time/Ts);  % Reference accelerations

%% Sliding Surface Parameters
% Gains for sliding surface definition (outer loop)
gama_a = [42 0 0;
          0 42 0;
          0 0 42];

gama_u = [0.1 0; 0 0]; % For load swing angles

% Sliding surface tuning parameters
ro_a = [48 0 0;
        0 48 0;
        0 0 48];

ro_u = 1*[.4 .2; .4 .2; .4 .2];

sigma = 2*eye(3); % Weighting matrix for sliding surface dynamics

%% Sliding Surface Definition
s = zeros(3,time/Ts);  % Sliding variable
sd = zeros(3,time/Ts); % Derivative of sliding variable
K_t = zeros(3,time/Ts); % Adaptive gain history

% Compute initial sliding surface (k=1)
s(:,1) = ro_a*((-qd_r(1:3,1)+qd(1:3,1)) + gama_a*(-q_r(1:3,1)+q(1:3,1))) ...
       + ro_u*((-qd_r(7:8,1)+qd(7:8,1)) + gama_u*(-q_r(7:8,1)+q(7:8,1)));
sd(:,1) = ro_a*((-qdd_r(1:3,1)+qdd(1:3,1)) + gama_a*(-qd_r(1:3,1)+qd(1:3,1))) ...
       + ro_u*((-qdd_r(7:8,1)+qdd(7:8,1)) + gama_u*(-qd_r(7:8,1)+qd(7:8,1)));

%% Adaptive Controller Parameters
% Initial control increments and adaptive gains
delta_u = [0; 0; 0];
delta_u2 = [0; 0; 0];
K_bar = [.01; .01; .01]; % Upper limit for K_dot growth
K = [.01; .01; .01];     % Initial adaptive gains
U_epsilon = [1; 1; 1];   % Discontinuous term
mu = [0.001;0.001;0.001]; % Small adaptation term
K_dot = zeros(3,1);      % Derivative of adaptive gain
mu_ep = [.2;.2;.2];      % Epsilon parameter

%% Simulation Loop
for k = 2:time/Ts-1

    % Estimated control effectiveness matrix (g_bar)
    b_hat = diag([250, 250, 250]);

    % Generate a simple reference trajectory (constant x,z; linearly increasing y)
    q_r(1:3,k+1)  = [0.2; k*0.001; 0.02];
    qd_r(1:3,k+1) = [0; 0.001; 0];

    % Update sliding surface and its derivative
    s(:,k) = ro_a*((-qd_r(1:3,k)+qd(1:3,k)) + gama_a*(-q_r(1:3,k)+q(1:3,k))) ...
           + ro_u*((-qd_r(7:8,k)+qd(7:8,k)) + gama_u*(-q_r(7:8,k)+q(7:8,k)));
    sd(:,k) = ro_a*((-qdd_r(1:3,k)+qdd(1:3,k)) + gama_a*(-qd_r(1:3,k)+qd(1:3,k))) ...
           + ro_u*((-qdd_r(7:8,k)+qdd(7:8,k)) + gama_u*(-qd_r(7:8,k)+qd(7:8,k)));

    % Discontinuous control law (sign of sliding surface)
    U_epsilon = -K.*sign(s(:,k)); 

    % Update adaptive gains K using a simple law:
    for i = 1:3
        if K(i) < mu(i)
            K_dot(i) = mu(i);
        else
            K_dot(i) = K_bar(i)*norm(s(i,k))*sign(norm(s(i,k))-mu_ep(i));
        end
    end
    K = K + K_dot*Ts; % Integrate K_dot
    K_t(:,k) = K;     % Store K

    % Outer-loop control increment (sliding mode control law)
    delta_u = b_hat \ (-sigma*s(:,k) - sd(:,k-1) + U_epsilon);
    delta_tau(:,k) = delta_u;

    % Update outer-loop torques
    tau(1:3,k) = tau(1:3,k-1) + delta_u;

    % Compute reference roll and pitch from force distribution
    phi_r = asin((tau(1,k)*sin(q(4,k)) - tau(2,k)*cos(q(4,k))) / ...
                sqrt(tau(1,k)^2 + tau(2,k)^2 + (tau(3,k)+ga)^2));
    theta_r = atan((tau(1,k)*cos(q(4,k)) + tau(2,k)*sin(q(4,k))) / ...
                (tau(3,k) + ga));
    q_r(4:6,k) = [0; theta_r; phi_r];

    % Inner-loop control increment (backstepping)
    delta_u2 = Backstepping_InnerLoop(q_r(:,k), qd_r(:,k), qdd_r(:,k), ...
                                      q(:,k), qd(:,k), qdd(:,k));
    tau(4:6,k) = tau(4:6,k-1) + delta_u2;

    % Hexacopter dynamic model update
    [qdd(:,k+1), u1] = HexacopterModel(q(:,k), qd(:,k), tau(:,k));

    % Integrate to get next velocity and position
    qd(:,k+1) = qd(:,k) + qdd(:,k+1)*Ts;        
    q(:,k+1) = q(:,k) + qd(:,k+1)*Ts;

    % Normalize attitude angles to [-pi, pi]
    q(4:6,k+1) = mod(q(4:6,k+1) + pi, 2*pi) - pi;

    % Ensure swing angle (alpha) is positive
    if q(7,k+1) < 0 
        q(7,k+1) = abs(q(7,k+1)); 
        qd(7,k+1) = -qd(7,k+1); 
        q(8,k+1) = q(8,k+1) + pi; % Flip azimuth by 180°
    end
    q(8,k+1) = mod(q(8,k+1) + pi, 2*pi) - pi;

    % Store swing angles in degrees for plotting
    swing(1,k+1) = rad2deg(q(7,k+1));
    swing(2,k+1) = rad2deg(q(8,k+1));

    % Recompute thrust contributions
    a1 = (cos(q(6,k+1))*sin(q(5,k+1))*cos(q(4,k+1)) + sin(q(6,k+1))*sin(q(4,k+1))) / (m+M);
    a2 = (cos(q(6,k+1))*sin(q(5,k+1))*sin(q(4,k+1)) - sin(q(6,k+1))*cos(q(4,k+1))) / (m+M);
    a3 = cos(q(6,k+1))*cos(q(5,k+1)) / (m+M);

    % Update outer-loop control torques for next step
    tau(1:3,k) = [a1*u1;
                  a2*u1; 
                  a3*u1 - ga];

    k % Display current iteration in command window
end

%% Plotting

% --- Position tracking and errors ---
figure(1)
subplot(3,2,1)
plot(Ts:Ts:time,q(1,:),'-'); hold on;
plot(Ts:Ts:time,q_r(1,:))
legend('Actual','Reference')
xlabel('Time (s)')
ylabel('x (m)')
title('x position')
grid minor

subplot(3,2,2)
plot(Ts:Ts:time,q(1,:)-q_r(1,:))
xlabel('Time (s)')
ylabel('e1 (m)')
title('Error in x position')
grid minor

subplot(3,2,3)
plot(Ts:Ts:time,q(2,:));hold on;
plot(Ts:Ts:time,q_r(2,:))
legend('Actual','Reference')
xlabel('Time (s)')
ylabel('y (m)') 
title('y position')
grid minor

subplot(3,2,4)
plot(Ts:Ts:time,q(2,:)-q_r(2,:))
xlabel('Time (s)')
ylabel('e2 (m)')
title('Error in y position')
grid minor

subplot(3,2,5)
plot(Ts:Ts:time,q(3,:));hold on;
plot(Ts:Ts:time,q_r(3,:))
legend('Actual','Reference')
xlabel('Time (s)')
ylabel('z (m)')
title('z position')
grid minor

subplot(3,2,6)
plot(Ts:Ts:time,q(3,:)-q_r(3,:))
xlabel('Time (s)')
ylabel('e3 (m)')
title('Error in z position')
grid minor

t = sgtitle('Position of the UAV and the Respective Errors');
t.FontSize = 12;          
t.FontWeight = 'bold';    
t.Color = 'blue';

% --- Swing angles of the load ---
figure(2)
subplot(2,1,1)
plot(Ts:Ts:time,swing(1,:));hold on;
plot(Ts:Ts:time,q_r(7,:))
legend('Actual','Reference')
xlabel('Time (s)')
ylabel('\alpha (degree)')
grid minor

subplot(2,1,2)
plot(Ts:Ts:time,swing(2,:));hold on;
plot(Ts:Ts:time,q_r(8,:))
legend('Actual','Reference')
xlabel('Time (s)')
ylabel('\beta (degree)')
grid minor

t = sgtitle('Swing Angles of the Load Attached to the UAV');
t.FontSize = 12;          
t.FontWeight = 'bold';    
t.Color = 'blue';

% --- Attitude angles of the UAV ---
figure(3)
subplot(3,1,1)
plot(Ts:Ts:time,q(4,:));hold on;
plot(Ts:Ts:time,q_r(4,:))
legend('Actual','Reference')
xlabel('Time (s)')
ylabel('\Psi (rad)')
grid minor

subplot(3,1,2)
plot(Ts:Ts:time,q(5,:));hold on;
plot(Ts:Ts:time,q_r(5,:))
legend('Actual','Reference')
xlabel('Time (s)')
ylabel('\Theta (rad)')
grid minor

subplot(3,1,3)
plot(Ts:Ts:time,q(6,:));hold on;
plot(Ts:Ts:time,q_r(6,:))
legend('Actual','Reference')
xlabel('Time (s)')
ylabel('\Phi (rad)')
grid minor

t = sgtitle('Attitude Angles of the UAV');
t.FontSize = 12;          
t.FontWeight = 'bold';    
t.Color = 'blue';