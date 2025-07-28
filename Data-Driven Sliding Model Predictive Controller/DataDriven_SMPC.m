clear all
clc
addpath('C:\Users\User\Desktop\Hexacopter paper\Thesis\casadi-3.6.5-windows64-matlab2018b')
import casadi.*

%% Parameters
Ts = 0.001; % sampling period [s]
time = 50;  % total simulation time [s]
ga = 9.81;  % gravity

% Initial states
q = zeros(8,time/Ts); 
x_r = 0.2;  % reference x
z_r = 0.02; % reference z
q(1,1) = x_r; 
q(2,1) = 0; 
q(3,1) = z_r;
q(7,1) = 0.0001; % initial small swing angle

qd  = zeros(8,time/Ts);
qdd = zeros(8,time/Ts);
u   = zeros(6,time/Ts);

% Reference trajectories
q_r = zeros(8,time/Ts);
q_r(1,1) = x_r;
q_r(3,1) = z_r;
qd_r = zeros(8,time/Ts);
qdd_r = zeros(8,time/Ts);

%% Errors
% Position error and derivatives
e_a  = zeros(3,time/Ts);
ed_a = zeros(3,time/Ts);
e_a(:,1)  = q(1:3,1) - q_r(1:3,1);
ed_a(:,1) = qd(1:3,1) - qd_r(1:3,1);

% Swing error and derivatives
e_u  = zeros(2,time/Ts);
ed_u = zeros(2,time/Ts);
e_u(:,1)  = q(7:8,1) - q_r(7:8,1);
ed_u(:,1) = qd(7:8,1) - qd_r(7:8,1);

%% Sliding Surface Parameters
lamda_a = diag([60 60 60]);
c_a     = diag([9 9 9]);
lamda_u = [1 .1; 1 .1; 1 .1];
c_u     = [15 0; 0 0];

s_a = zeros(3,time/Ts);
s_u = zeros(3,time/Ts);
s   = zeros(3,time/Ts);

s_a(:,1) = lamda_a*(ed_a(:,1) + c_a*e_a(:,1));
s_u(:,1) = lamda_u*(ed_u(:,1) + c_u*e_u(:,1));
s(:,1)   = s_a(:,1) + s_u(:,1);

%% Linear Approximation (for MPC)
gbar = 0.010;
X = zeros(6,time/Ts);

A = [diag([2 2 2]) diag([-1 -1 -1]); 
     diag([1 1 1]) zeros(3,3)];
B = [diag([gbar*Ts, gbar*Ts, gbar*Ts]);
     zeros(3,3)];

%% MPC Parameters
N = 20; % prediction horizon
Q = diag([25 25 25 15 15 15]); % state weights
R = diag([1 1 1]);             % input weights

%% CasADi symbolic MPC model
s21 = SX.sym('s21'); s22 = SX.sym('s22'); s23 = SX.sym('s23');
s11 = SX.sym('s11'); s12 = SX.sym('s12'); s13 = SX.sym('s13');
states_sym = [s21; s22; s23; s11; s12; s13]; n_states = length(states_sym);

delta_ux = SX.sym('delta_ux');
delta_uy = SX.sym('delta_uy');
delta_uz = SX.sym('delta_uz');
delta_u_sym = [delta_ux; delta_uy; delta_uz]; n_inputs = length(delta_u_sym);

rhs = A*states_sym + B*delta_u_sym;
f = Function('f',{states_sym,delta_u_sym},{rhs});

X_sym = SX.sym('X',n_states,N+1);
U_sym = SX.sym('U',n_inputs,N);
P_sym = SX.sym('P',2*n_states);

X_sym(:,1) = P_sym(1:n_states);
for k=1:N
    X_sym(:,k+1) = f(X_sym(:,k),U_sym(:,k));
end

% Cost function
obj = 0;
for k = 1:N
    obj = obj + (X_sym(:,k)-P_sym(n_states+1:2*n_states))'*Q*(X_sym(:,k)-P_sym(n_states+1:2*n_states)) + ...
                U_sym(:,k)'*R*U_sym(:,k);
end

% Constraints: all predicted states
g = [];
for k = 1:N+1
    g = [g;X_sym(1:n_states,k)];
end

OPT_variables = reshape(U_sym,n_inputs*N,1);
nlp_prob = struct('f',obj,'x',OPT_variables,'g',g,'p',P_sym);

% Solver options
opts = struct;
opts.ipopt.max_iter = 100;
opts.ipopt.print_level = 0;
opts.print_time = 0;
opts.ipopt.acceptable_tol = 1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;
solver = nlpsol('solver','ipopt',nlp_prob,opts);

args = struct;
% State constraints (no limits here)
args.lbg = -inf*ones(size(g));
args.ubg =  inf*ones(size(g));
% Input constraints
args.lbx = -inf;
args.ubx =  inf;

u0 = zeros(3,N);
X(:,1) = [s(:,1); zeros(length(s(:,1)),1)];
swing = zeros(2,time/Ts);

%% Simulation loop
for i=1:time/Ts-1
    i  % show current step

    x0 = X(:,i);
    xs = zeros(n_states,1);
    args.p = [x0;xs];
    args.x0 = reshape(u0,3*N,1);
    
    sol = solver('x0',args.x0,'lbx',args.lbx,'ubx',args.ubx,...
                 'lbg',args.lbg,'ubg',args.ubg,'p',args.p);
    delta_u_star = reshape(full(sol.x)',n_inputs,N);
    u0 = [delta_u_star(:,2:N), delta_u_star(:,N)];
    delta_u = delta_u_star(:,1);

    % Update control inputs (outer loop)
    if i == 1
        u(1:3,i) = delta_u;
    else
        u(1:3,i) = u(1:3,i-1) + delta_u;
    end

    % Reference angles from outer loop forces
    phi_r = asin((u(1,i)*sin(q(4,i)) - u(2,i)*cos(q(4,i))) / ...
                  sqrt(u(1,i)^2 + u(2,i)^2 + (u(3,i)+ga)^2));
    theta_r = atan((u(1,i)*cos(q(4,i)) + u(2,i)*sin(q(4,i))) / (u(3,i)+ga));
    q_r(4:6,i) = [0; theta_r; phi_r];

    % Inner loop control
    delta_u2 = Backstepping_InnerLoop(q_r(:,i), qd_r(:,i), qdd_r(:,i), q(:,i), qd(:,i), qdd(:,i));
    if i == 1
        u(4:6,i) = delta_u2;
    else
        u(4:6,i) = u(4:6,i-1) + delta_u2;
    end

    % UAV dynamics
    [qdd(:,i+1), ~] = HexacopterModel(q(:,i),qd(:,i),u(:,i));

    % Integrate states
    qd(:,i+1) = qd(:,i) + qdd(:,i+1)*Ts;        
    q(:,i+1)  = q(:,i) + qd(:,i+1)*Ts;

    % Normalize attitude
    q(4:6,i+1) = mod(q(4:6,i+1)+pi, 2*pi) - pi;

    % Flip swing angles if negative (cable crossing vertical)
    if q(7,i+1) < 0
        q(7,i+1) = abs(q(7,i+1)); 
        qd(7,i+1) = -qd(7,i+1); 
        q(8,i+1) = q(8,i+1) + pi;
    end
    q(8,i+1) = mod(q(8,i+1)+pi, 2*pi) - pi;

    swing(1,i+1) = rad2deg(q(7,i+1));
    swing(2,i+1) = rad2deg(q(8,i+1));

    % Reference trajectory update (constant x, z; moving y)
    q_r(1:3,i+1)  = [x_r; i*0.001; z_r];
    qd_r(1:3,i+1) = [0; 0.001; 0];
    qdd_r(1:3,i+1) = [0; 0; 0];

    % Update sliding surface
    e_a(:,i+1)  = q(1:3,i+1) - q_r(1:3,i+1);
    e_u(:,i+1)  = q(7:8,i+1) - q_r(7:8,i+1);
    ed_a(:,i+1) = qd(1:3,i+1) - qd_r(1:3,i+1);
    ed_u(:,i+1) = qd(7:8,i+1) - qd_r(7:8,i+1);

    s_a(:,i+1) = lamda_a*(ed_a(:,i+1) + c_a*e_a(:,i+1));
    s_u(:,i+1) = lamda_u*(ed_u(:,i+1) + c_u*e_u(:,i+1));
    s(:,i+1)   = s_a(:,i+1) + s_u(:,i+1);

    X(:,i+1) = [s(:,i+1); s(:,i)];
end

save('smpc_alpha.mat', 'swing')

%% Plotting

% --- Position tracking ---
figure(1)
subplot(3,2,1)
plot(Ts:Ts:time,q(1,:),'-'); hold on; plot(Ts:Ts:time,q_r(1,:))
legend('Actual','Reference'); xlabel('Time (s)'); ylabel('x (m)'); title('x position'); grid on

subplot(3,2,2)
plot(Ts:Ts:time,q(1,:)-q_r(1,:))
xlabel('Time (s)'); ylabel('e1 (m)'); title('Error in x position'); grid on

subplot(3,2,3)
plot(Ts:Ts:time,q(2,:),'LineWidth',1.3); hold on; plot(Ts:Ts:time,q_r(2,:))
legend('Actual','Reference'); xlabel('Time (s)'); ylabel('y (m)'); title('y position'); grid on

subplot(3,2,4)
plot(Ts:Ts:time,q(2,:)-q_r(2,:))
xlabel('Time (s)'); ylabel('e2 (m)'); title('Error in y position'); grid on

subplot(3,2,5)
plot(Ts:Ts:time,q(3,:)); hold on; plot(Ts:Ts:time,q_r(3,:))
legend('Actual','Reference'); xlabel('Time (s)'); ylabel('z (m)'); title('z position'); grid on

subplot(3,2,6)
plot(Ts:Ts:time,q(3,:)-q_r(3,:))
xlabel('Time (s)'); ylabel('e3 (m)'); title('Error in z position'); grid on

t = sgtitle('Position of the UAV and the Respective Errors');
t.FontSize = 12; t.FontWeight = 'bold'; t.Color = 'blue';

% --- Swing angles ---
figure(2)
subplot(2,1,1)
plot(Ts:Ts:time,swing(1,:)); hold on; plot(Ts:Ts:time,q_r(7,:))
legend('Actual','Reference'); xlabel('Time (s)'); ylabel('\alpha (degree)'); grid on

subplot(2,1,2)
plot(Ts:Ts:time,swing(2,:)); hold on; plot(Ts:Ts:time,q_r(8,:))
legend('Actual','Reference'); xlabel('Time (s)'); ylabel('\beta (degree)'); grid on

t = sgtitle('Swing Angles of the Load Attached to the UAV');
t.FontSize = 12; t.FontWeight = 'bold'; t.Color = 'blue';

% --- Attitude angles ---
figure(3)
subplot(3,1,1)
plot(Ts:Ts:time,q(4,:)); hold on; plot(Ts:Ts:time,q_r(4,:))
legend('Actual','Reference'); xlabel('Time (s)'); ylabel('\Psi (rad)'); grid on

subplot(3,1,2)
plot(Ts:Ts:time,q(5,:)); hold on; plot(Ts:Ts:time,q_r(5,:))
legend('Actual','Reference'); xlabel('Time (s)'); ylabel('\Theta (rad)'); grid on

subplot(3,1,3)
plot(Ts:Ts:time,q(6,:)); hold on; plot(Ts:Ts:time,q_r(6,:))
legend('Actual','Reference'); xlabel('Time (s)'); ylabel('\Phi'); grid on

t = sgtitle('Attitude Angles of the UAV');
t.FontSize = 12; t.FontWeight = 'bold'; t.Color = 'blue';