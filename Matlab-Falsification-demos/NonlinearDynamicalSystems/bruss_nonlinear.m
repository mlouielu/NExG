% This is demo file for Example 3.1 in the technical report:
% "Probabilistic Temporal Logic Falsification of Cyber-Physical Systems" 
% H. Abbas, G. Fainekos, S. Sankaranarayanan, F. Ivancic, and A. Gupta
%
% This example also demonstrates the difference between modeling 
% requirements as a polyhedral set or a logical conjunction of halfspaces.

% (C) Georgios Fainekos 2011 - Arizona State University
max_iter = 1;

iterations = zeros(1,max_iter);
robustness = zeros(1, max_iter);

idx = 1;
while idx <= max_iter

%      clear

%     cd('../..')
%     cd('SystemModelsAndData')

    % disp(' ')
    % disp(' Demo: Simulated Annealing on the Example 3.1 from TECS 2013 paper. ')
    % disp(' One run will be performed for a maximum of 1000 tests. ')
    % disp(' Press any key to continue ... ')

    % pause

    model = @(t,x) ...
        [ 1 + x(1) * x(1) * x(2) - 1.5 * x(1) - x(1); ...
        1.5 * x(1) - x(1)* x(1)* x(2)];

    init_cond = [0.2 1.5; 1.1 1.4];
    input_range = [];
    cp_array = [];

    phi{1} = '[]!a';
    phi{2} = '[]!(a1 /\ a2 /\ a3 /\ a4)';
    % 
    % disp(' ')
    % disp(' Select a requirement: ')
    % disp('    1. []!a where O(a) = [-1.6,-1.4]x[-1.1,-.9] ')
    % disp('       (slow robustness computation; quadratic optimization programs must be solved) ')
    % disp('    2. []!(a1 /\ a2 /\ a3 /\ a4) where a1 = x1>=-1.6, a2 = x1<=-1.4,  a3 = x2>=-1.1, a4 = x1<=-0.9')
    % disp('       (fast robustness computation; analytical distance computations) ')
    % disp(' ')
    % form_id = input ('Select an option (1-2): ');

    form_id = 1;

    u_x_min = 0.58;
    u_x_max = 0.61;
    u_y_min = 1.7;
    u_y_max = 1.72;
    ii = 1;
    preds(ii).str='a';
    preds(ii).A = [1 0; 1 0; 0 1; 0 1];
    preds(ii).b = [u_x_min; u_x_max; u_y_min; u_y_max];
    ii = ii+1;
    preds(ii).str='a1';
    preds(ii).A = [1 0];
    preds(ii).b = u_x_min;
    ii = ii+1;
    preds(ii).str='a2';
    preds(ii).A = [1 0];
    preds(ii).b = u_x_max;
    ii = ii+1;
    preds(ii).str='a3';
    preds(ii).A = [0 1];
    preds(ii).b = u_y_min;
    ii = ii+1;
    preds(ii).str='a4';
    preds(ii).A = [0 1];
    preds(ii).b = u_y_max;

    time = 2;

    opt = staliro_options();

    opt.runs = 1;

    % Since this a function pointer there is no output space.
    % Set the specification space to be X
    opt.spec_space = 'X';

    % This model needs a stiff solver
    opt.ode_solver = 'ode15s';

    % Set the max number of tests
    opt.optim_params.n_tests = 9;

    [results, history] = staliro(model,init_cond,input_range,cp_array,phi{form_id},preds,time,opt);
    
    if results.run.nTests > 1 && results.run.nTests < 100
        iterations(idx) = results.run.nTests;
        robustness(idx) = results.run.bestRob;
        idx = idx + 1;
    end

    % Get the falsifying trajectory
    bestRun = results.optRobIndex;
    [T1,XT1] = SimFunctionMdl(model,init_cond,input_range,cp_array,results.run(bestRun).bestSample,time,opt);

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
