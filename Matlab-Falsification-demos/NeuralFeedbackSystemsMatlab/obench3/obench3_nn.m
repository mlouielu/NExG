max_iter = 250;

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

%     model = @(t,x) ...
%         [ x(2); ...
%         x(2) - x(1)* x(1)* x(2) - x(1)];

    model = @obench3ODE;

    init_cond = [0.5 1.2; 0.5 1.2];
    input_range = [];
    cp_array = [];

%     phi{1} = '[]!a';
%     phi{2} = '[]!(a1 /\ a2 /\ a3 /\ a4)';
    phi = '!(<>_[0.8,1.0] eventual_req)';
    
%     phi = '!([]_[0.8,1.0] always_req)';

    form_id = 1;

    u_x_min = 0.3;
    u_x_max = 0.4;
    u_y_min = 0.00;
    u_y_max = 0.1;
    ii = 1;
    preds(ii).str='eventual_req';
    preds(ii).A = [-1 0; 1 0; 0 -1; 0 1];
    preds(ii).b = [-u_x_min; u_x_max; -u_y_min; u_y_max];
    
%     u_x_min = 0.4;
%     u_x_max = 0.5;
%     u_y_min = -0.1;
%     u_y_max = 0.0;
%     ii = 1;
%     preds(ii).str='always_req';
%     preds(ii).A = [-1 0; 1 0; 0 -1; 0 1];
%     preds(ii).b = [-u_x_min; u_x_max; -u_y_min; u_y_max];
%     
%     ii = ii+1;
%     preds(ii).str='a1';
%     preds(ii).A = [-1 0];
%     preds(ii).b = -u_x_min;
%     ii = ii+1;
%     preds(ii).str='a2';
%     preds(ii).A = [1 0];
%     preds(ii).b = u_x_max;
%     ii = ii+1;
%     preds(ii).str='a3';
%     preds(ii).A = [0 -1];
%     preds(ii).b = -u_y_min;
%     ii = ii+1;
%     preds(ii).str='a4';
%     preds(ii).A = [0 1];
%     preds(ii).b = u_y_max;

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
    
    if results.run.nTests > 1
        iterations(idx) = results.run.nTests;
        robustness(idx) = results.run.bestRob;
        idx = idx + 1;
        idx
    end

    % Get the falsifying trajectory
%     bestRun = results.optRobIndex;
%     [T1,XT1] = SimFunctionMdl(model,init_cond,input_range,cp_array,results.run(bestRun).bestSample,time,opt);

%     figure(1)
%     clf
%     rectangle('Position',[u_x_min,u_y_min,u_x_max-u_x_min,u_y_max-u_y_min],'FaceColor','r')
%     hold on
%     if (init_cond(1,1)==init_cond(1,2)) || (init_cond(2,1)==init_cond(2,2))
%         plot(init_cond(1,:),init_cond(2,:),'g')
%     else
%         rectangle('Position',[init_cond(1,1),init_cond(2,1),init_cond(1,2)-init_cond(1,1),init_cond(2,2)-init_cond(2,1)],'FaceColor','g')
%     end
%     ntests = results.run(bestRun).nTests;
%     hist = history(bestRun).samples;
%     plot(hist(1:ntests,1),hist(1:ntests,2),'*')
%     plot(XT1(:,1),XT1(:,2))
%     xlabel('y_1')
%     ylabel('y_2')

%     cd('..')
%     cd('Falsification demos/HSCC2021')
end

mean_iter = mean(iterations);
var_iter = var(iterations);
maximum_iterations = max(iterations);
mean_robust = mean(robustness);
