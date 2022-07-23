max_iter = 1;

iterations = zeros(1,max_iter);
robustness = zeros(1, max_iter);

idx = 1;
while idx <= max_iter

 %     clear

%     cd('../..')
%     cd('SystemModelsAndData')

    % disp(' ')
    % disp(' One run will be performed for a maximum of 1000 tests. ')
    % disp(' Press any key to continue ... ')

    % pause

    model = 'RL_CARLA_BRAKE_SYSTEM';
    ss = 'RL_CARLA_BRAKE_SYSTEM/Discrete State-Space';
    
    init_cond = [0.5 1.5; 0.5 0.5; 0.5 0.5]; %works for comparison with staliro
    input_range = [];
    cp_array = [];
    
    X0 = init_cond;
    
    phi = '!(<>_[0.70,0.90] eventual_req)';

    form_id = 1;

    u_x_min = 1.0;
    u_x_max = 1.05;
    u_y_min = 0.05;
    u_y_max = 0.1;
    u_z_min = -1.15;
    u_z_max = -1.10;
    ii = 1;
    preds(ii).str='eventual_req';
    preds(ii).A = [-1 0 0; 1 0 0; 0 -1 0; 0 1 0; 0 0 -1; 0 0 1];
    preds(ii).b = [-u_x_min; u_x_max; -u_y_min; u_y_max; -u_z_min; u_z_max];
    
    time = 2.0;

    opt = staliro_options();

    opt.runs = 1;

    % Since this a function pointer there is no output space.
    % Set the specification space to be X
    opt.spec_space = 'X';

    % This model needs a stiff solver
%     opt.ode_solver = 'ode15s';

    % Set the max number of tests
    opt.optim_params.n_tests = 50;

    [results, history] = staliro(model,init_cond,input_range,cp_array,phi,preds,time,opt);
    
    if results.run.nTests > 1
        iterations(idx) = results.run.nTests;
        robustness(idx) = results.run.bestRob;
        idx = idx + 1;
        idx

%         % Get the falsifying trajectory
%         bestRun = results.optRobIndex;
%         [T1,XT1] = SimFunctionMdl(model,init_cond,input_range,cp_array,results.run(bestRun).bestSample,time,opt);
% 
%         figure(1)
%         clf
%         rectangle('Position',[u_x_min,u_y_min,u_x_max-u_x_min,u_y_max-u_y_min],'FaceColor','r')
%         hold on
%         if (init_cond(1,1)==init_cond(1,2)) || (init_cond(2,1)==init_cond(2,2))
%             plot(init_cond(1,:),init_cond(2,:),'g')
%         else
%             rectangle('Position',[init_cond(1,1),init_cond(2,1),init_cond(1,2)-init_cond(1,1),init_cond(2,2)-init_cond(2,1)],'FaceColor','g')
%         end
%         ntests = results.run(bestRun).nTests;
%         hist = history(bestRun).samples;
%         plot(hist(1:ntests,1),hist(1:ntests,2),'*')
%         plot(XT1(:,1),XT1(:,2))
%         xlabel('y_1')
%         ylabel('y_2')
    end
end

mean_iter = mean(iterations);
var_iter = var(iterations);
maximum_iterations = max(iterations);
mean_robust = mean(robustness);

