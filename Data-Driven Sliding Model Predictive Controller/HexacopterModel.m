function [qdd, u1] = HexacopterModel(q, q_d, tau)
% HEXACOPTERMODEL 
% Computes the state accelerations (qdd) of a hexacopter carrying a 
% suspended payload. 
% This is a coupled nonlinear model of a UAV and a cable-suspended load.
%
% Inputs:
%   q     - State vector [x; y; z; psi; theta; phi; alpha; beta]
%           Positions, attitudes (yaw, pitch, roll) and swing angles
%   q_d   - First derivative of q (velocities)
%   tau   - Control vector [tau_x; tau_y; tau_z; tau_psi; tau_theta; tau_phi]
%           The first 3 elements correspond to translational forces in
%           body frame, the last 3 are torques/commands.
%
% Outputs:
%   qdd   - Second derivative of q (accelerations)
%   u1    - Collective thrust (total upward force)

% ----------------------------
% Parameters
% ----------------------------
M = 0.4;      % UAV mass (kg)
m = 0.03;     % Suspended load mass (kg)
g = 9.81;     % Gravitational acceleration (m/s^2)
d = 0.1;      % Arm offset (not used here)
l = 0.35;     % Cable length (m)
I_p = 1.00e-6; % Payload inertia about pivot
I_x = 1.77e-3; % UAV inertia around x-axis
I_y = 1.77e-3; % UAV inertia around y-axis
I_z = 3.54e-3; % UAV inertia around z-axis

% ----------------------------
% Compute collective thrust u1
% ----------------------------
% u1 represents the magnitude of the thrust vector required to produce 
% the net accelerations commanded by tau(1:3)
u1 = (M+m)*sqrt( tau(1)^2 + tau(2)^2 + (tau(3)+g)^2 );

% Full control input vector: [total thrust; roll torque; pitch torque; yaw torque]
u = [u1; tau(4); tau(5); tau(6)];

% ----------------------------
% Extract states from q
% ----------------------------
x = q(1);       y = q(2);       z = q(3);
psi = q(4);     theta = q(5);   phi = q(6);
alpha = q(7);   beta = q(8);

% Extract derivatives from q_d
x_d = q_d(1);       y_d = q_d(2);       z_d = q_d(3);
psi_d = q_d(4);     theta_d = q_d(5);   phi_d = q_d(6);
alpha_d = q_d(7);   beta_d = q_d(8);

% ----------------------------
% Mass/Inertia Matrix M(q)
% ----------------------------
% This matrix represents the coupled dynamics between UAV and payload.
% Diagonal terms: mass/inertia
% Off-diagonal terms: coupling between the UAV and the swinging payload
m11 = M+m; m22 = m11; m33 = m11;
m17 = m*l*cos(alpha)*cos(beta);
m18 = -m*l*sin(alpha)*sin(beta);
m27 = m*l*cos(alpha)*sin(beta);
m28 = m*l*sin(alpha)*cos(beta);
m37 = m*l*sin(alpha);
m44 = I_x*sin(theta)^2 + cos(theta)^2*(I_y*sin(phi)^2 + I_z*cos(phi)^2);
m45 = (I_y-I_z)*(cos(theta)*sin(phi)*cos(phi));
m55 = I_y*cos(phi)^2 + I_z*sin(phi)^2;
m77 = m*l^2 + I_p;
m88 = m*l^2*sin(alpha)^2 + I_p;

M_q = [m11 0 0 0 0 0 m17 m18;
       0 m22 0 0 0 0 m27 m28;
       0 0 m33 0 0 0 m37 0;
       0 0 0 m44 m45 -I_x*sin(theta) 0 0;
       0 0 0 m45 m55 0 0 0;
       0 0 0 -I_x*sin(theta) 0 I_x 0 0;
       m17 m27 m37 0 0 0 m77 0;
       m18 m28 0 0 0 0 0 m88];

% ----------------------------
% Coriolis/Centrifugal Matrix C(q)
% ----------------------------
% These terms include coupling due to motion of the payload and 
% angular velocities of the UAV
c17 = -m*l*(cos(alpha)*sin(beta)*beta_d + sin(alpha)*cos(beta)*alpha_d);
c18 = -m*l*(cos(alpha)*sin(beta)*alpha_d + sin(alpha)*cos(beta)*beta_d);
c27 = m*l*(cos(alpha)*cos(beta)*beta_d - sin(alpha)*sin(beta)*alpha_d);
c28 = m*l*(cos(alpha)*cos(beta)*alpha_d - sin(alpha)*sin(beta)*beta_d);
% Many terms correspond to the nonlinear rotational dynamics (see Euler-Lagrange)
c44 = I_x*theta_d*sin(theta)*cos(theta) ...
       - (I_y+I_z)*(theta_d*sin(theta)*cos(theta)*sin(phi)^2) ...
       + (I_y-I_z)*phi_d*cos(theta)^2*sin(phi)*cos(phi);
c45 = I_x*psi_d*sin(theta)*cos(theta) ...
       - (I_y-I_z)*(theta_d*sin(theta)*cos(phi)*sin(phi) + phi_d*cos(theta)*sin(phi)^2) ...
       - (I_y+I_z)*(psi_d*sin(theta)*cos(theta)*cos(phi)^2 - phi_d*cos(theta)*cos(phi)^2);
c46 = -(I_x*theta_d*cos(theta) - (I_y-I_z)*(psi_d*cos(theta)^2*sin(phi)*cos(phi)));
c54 = psi_d*sin(theta)*cos(theta)*(-I_x+I_y*sin(phi)^2+I_z*cos(phi)^2);
c55 = -(I_y-I_z)*(phi_d*sin(phi)*cos(phi));
c56 = I_x*psi_d*cos(theta) ...
       + (I_y-I_z)*(-theta_d*sin(theta)*cos(phi) + psi_d*cos(theta)*(cos(phi)^2 - sin(phi)^2));
c64 = -(I_y-I_z)*(psi_d*cos(theta)^2*sin(phi)*cos(phi));
c65 = -I_x*psi_d*cos(theta) ...
       + (I_y-I_z)*(theta_d*sin(phi)*cos(phi) + psi_d*cos(theta)*(sin(phi)^2 - cos(phi)^2));

C_q = [0 0 0 0 0 0 c17 c18;
       0 0 0 0 0 0 c27 c28;
       0 0 0 0 0 0 m*l*cos(alpha)*alpha_d 0;
       0 0 0 c44 c45 c46 0 0;
       0 0 0 c54 c55 c56 0 0;
       0 0 0 c64 c65 0 0 0;
       0 0 0 0 0 0 0 -m*l^2*sin(alpha)*cos(alpha)*beta_d;
       0 0 0 0 0 0 m*l^2*sin(alpha)*cos(alpha)*beta_d m*l^2*sin(alpha)*cos(alpha)*alpha_d];

% ----------------------------
% Gravity Vector G(q)
% ----------------------------
% Gravitational forces acting on UAV and load
G_q = [0;
       0;
       (M+m)*g;
       0;
       0;
       0;
       m*l*g*sin(alpha);
       0];

% ----------------------------
% Input distribution matrix b
% ----------------------------
% Relates control inputs u to generalized forces
b = [sin(phi)*sin(psi)+cos(phi)*cos(psi)*sin(theta) 0 0 0;
     cos(phi)*sin(theta)*sin(psi)-cos(psi)*sin(phi) 0 0 0;
     cos(theta)*cos(phi) 0 0 0;
     0 1 0 0;
     0 0 1 0;
     0 0 0 1;
     0 0 0 0;
     0 0 0 0];

% ----------------------------
% System dynamics (Euler-Lagrange)
% ----------------------------
% M(q)*qdd + C(q)*q_d + G(q) = b*u
% => qdd = inv(M(q)) * (-C(q)*q_d - G(q) + b*u)
qdd = M_q \ (- C_q*q_d - G_q + b*u);

end