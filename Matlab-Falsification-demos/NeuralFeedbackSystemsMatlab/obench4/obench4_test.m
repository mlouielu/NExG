model = @obench4ODE;

init_cond = [0.3 0.6; 0.3 0.6; 0.3 0.6];
input_range = [];
cp_array = [];

phi = '!([]_[0.7,0.9] always_req)';

form_id = 1;

u_x_min = 0.7;
u_x_max = 0.8;
u_y_min = -0.1;
u_y_max = 0.0;
u_z_min = -0.9;
u_z_max = -0.8;
ii = 1;
preds(ii).str='always_req';
preds(ii).A = [-1 0 0; 1 0 0; 0 -1 0; 0 1 0; 0 0 -1; 0 0 1];
preds(ii).b = [-u_x_min; u_x_max; -u_y_min; u_y_max; -u_z_min; u_z_max];

time = 2.0;

opt = staliro_options();

opt.runs = 1;

% Since this a function pointer there is no output space.
% Set the specification space to be X
opt.spec_space = 'X';

% This model needs a stiff solver
opt.ode_solver = 'ode15s';

% Set the max number of tests
opt.optim_params.n_tests = 50;

[results, history] = staliro(model,init_cond,input_range,cp_array,phi,preds,time,opt);
 
% Get the falsifying trajectory
bestRun = results.optRobIndex;
[T1,XT1] = SimFunctionMdl(model,init_cond,input_range,cp_array,results.run(bestRun).bestSample,time,opt);
% 
%         figure(1)
%         clf
%         rectangle('Position',[u_x_min,u_y_min,u_x_max-u_x_min,u_y_max-u_y_min],'FaceColor','r')
%         hold on
%         if (init_cond(1,1)==init_cond(1,2)) || (init_cond(2,1)==init_cond(2,2))
%             plot(init_cond(1,:),init_cond(2,:),'g')
%         else
%             rectangle('Position',[init_cond(1,1),init_cond(2,1),init_cond(1,2)-init_cond(1,1),init_cond(2,2)-init_cond(2,1)], 'LineWidth', 1.5)
%         end
%         ntests = results.run(bestRun).nTests;
%         hist = history(bestRun).samples;
%         plot(hist(1:ntests,1),hist(1:ntests,2),'x', 'Color', 'b')
%         plot(XT1(:,1),XT1(:,2), 'LineWidth', 1.5)
%         xlabel('y_1')
%         ylabel('y_2')

