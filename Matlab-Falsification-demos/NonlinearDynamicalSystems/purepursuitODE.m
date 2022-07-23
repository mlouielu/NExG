function [ret]=purepursuitODE(T,X)

%PUREPURSUITEQ Summary of this function goes here
%   Detailed explanation goes here

%parameters
s = 1.0; % Speed of the rear axel in meters/second
l = 1.2; % Lookahead distance in meters
L = 0.33; % Wheelbase of the vehicle

% Name input variables
% (a,b): coordinates of the first endpoint of the line segment
% (c,d): coordinates of the second endpoint of the line segment

a = 0;
b = 0; % (a,b): coordinates of the first endpoint of the line segment
c = 12;
d = 0; % (c,d): coordinates of the second endpoint of the line segment

% Name state variables
x = X(1);
y = X(2);
theta = X(3);

xa = x-a;
yb = y-b;
ca = c-a;
db = d-b;
ca2 = ca^2;
db2 = db^2;
l2 = l^2;

Delta = (xa*ca+yb*db)^2-(ca2+db2)*(xa^2+yb^2-l2);
T1 = (xa*ca+yb*db+sqrt(Delta))/(ca2+db2);
xdot=s*cos(theta);
ydot=s*sin(theta);
x0 = a + T1*(ca);
y0 = b + T1*(db);
y0p = -(x0-x)*sin(theta)+(y0-y)*cos(theta);

ret = [xdot;
        ydot;
        2*L*y0p/l2];