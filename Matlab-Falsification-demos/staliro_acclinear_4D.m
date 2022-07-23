clear

cd('..')
cd('SystemModelsAndData')

disp(' ')
disp(' Demo: Simulated Annealing on the Example 3.1 from TECS 2013 paper. ')
disp(' One run will be performed for a maximum of 1000 tests. ')
disp(' Press any key to continue ... ')

pause

model = @(t,x) ...
    [ -x(2) + x(3); ...
    x(4); ...
    0; ...
    x(1) - 4*x(2)+3*x(3)-1.2*x(4)-10];

init_cond = [2.0 5.0; 18.0 22.0; 20.0 20.0; -1.0 1.0];
input_range = [];
cp_array = [];

phi{1} = '[]!a';
form_id = 1;

u_x_min = 0.0; 
u_x_max = 1.95; 
u_y_min = 10; 
u_y_max = 22; 
% u_z_min = -1.6; 
% u_z_max = -1.55;
% u_w_min = 32.1; 
% u_w_max = 32.2;
ii = 1;
preds(ii).str='a';
% preds(ii).A = [-1 0 0 0; 1 0 0 0; 0 -1 0 0; 0 1 0 0; 0 0 -1 0; 0 0 1 0; 0 0 0 -1; 0 0 0 1];
% preds(ii).b = [-u_x_min; u_x_max; -u_y_min; u_y_max; -u_z_min; u_z_max; -u_w_min; u_w_max];
preds(ii).A = [-1 0 0 0; 1 0 0 0; 0 -1 0 0; 0 1 0 0];
preds(ii).b = [-u_x_min; u_x_max; -u_y_min; u_y_max];
    
time = 6.0;

opt = staliro_options();

opt.runs = 1;

% Since this a function pointer there is no output space.
% Set the specification space to be X
opt.spec_space = 'X';

% This model needs a stiff solver
opt.ode_solver = 'ode15s';

% Set the max number of tests
opt.optim_params.n_tests = 100;

[results, history] = staliro(model,init_cond,input_range,cp_array,phi{form_id},preds,time,opt);

% Get the falsifying trajectory
bestRun = results.optRobIndex;
[T1,XT1] = SimFunctionMdl(model,init_cond,input_range,cp_array,results.run(bestRun).bestSample,time,opt);

figure(1)
clf
rectangle('Position',[u_x_min,u_y_min,u_x_max-u_x_min,u_y_max-u_y_min],'FaceColor','r')
hold on
if (init_cond(1,1)==init_cond(1,2)) || (init_cond(2,1)==init_cond(2,2))
    plot(init_cond(1,:),init_cond(2,:),'g')
else
    rectangle('Position',[init_cond(1,1),init_cond(2,1),init_cond(1,2)-init_cond(1,1),init_cond(2,2)-init_cond(2,1)],'FaceColor','g')
end
ntests = results.run(bestRun).nTests;
hist = history(bestRun).samples;
plot(hist(1:ntests,1),hist(1:ntests,2),'*')
plot(XT1(:,1),XT1(:,2))
xlabel('s')
ylabel('v')

cd('..')
cd('Falsification demos')