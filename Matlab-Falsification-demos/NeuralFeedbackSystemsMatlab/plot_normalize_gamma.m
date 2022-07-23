% Normalize first and then take mean

gamma_net1 = [1.87 1.81 1.76 1.66; 
        5.34 4.66 4.21 3.59;
        3.41 3.18 3.0 2.68;
        1.35 1.32 1.29 1.22;
        1.18 1.16 1.14 1.1;
        3.71 3.45 3.25 3.11;
        3.17 3.02 2.87 2.77;
        6.36 5.74 5.06 4.84;
        2.46 2.38 2.31 2.19;
        0.73 0.70 0.69 0.66;
        1.18 1.15 1.13 1.09;
        2.00 1.93 1.86 1.82;
        2.14 2.07 2.0 1.88;
        2.19 2.11 2.04 1.92;
        3.17 2.96 2.78 2.49;
        8.38 6.77 5.78 5.36;
        0.67 0.67 0.66 0.65;
        0.35 0.35 0.34 0.34;
        0.83 0.81 0.80 0.77;
        0.83 0.82 0.81 0.79];
    
gamma_net2 = [2.31 2.23 2.16 2.03;
        2.65 2.53 2.43 2.24;
        3.75 3.47 3.24 2.86;
        1.23 1.2 1.17 1.12;
        0.81 0.8 0.79 0.77;
        3.1 2.92 2.77 2.66;
        2.94 2.79 2.65 2.55;
        5.73 5.14 4.68 4.50;
        2.21 2.11 2.01 1.84;
        0.70 0.67 0.66 0.63;
        1.66 1.61 1.56 1.47;
        2.87 2.68 2.53 2.48;
        1.75 1.7 1.65 1.57;
        2.53 2.41 2.31 2.12;
        2.24 2.15 2.06 1.92;
        5.89 5.26 4.74 4.42;
        0.57 0.56 0.56 0.56;
        0.30 0.30 0.30 0.29;
        0.33 0.33 0.32 0.32;
        0.66 0.66 0.65 0.64];
    
gamma_net3 = [1.13 1.11 1.08 1.04;
        2.21 2.14 2.07 1.94;
        0.73 0.71 0.70 0.67;
        1.14 1.11 1.09 1.04;
        1.39 1.38 1.36 1.32;
        3.30 3.09 2.93 2.81;
        3.81 3.57 3.36 3.23;
        3.16 3.0 2.84 2.72;
        1.15 1.13 1.11 1.06;
        0.90 0.88 0.86 0.82;
        0.51 0.51 0.51 0.50;
        0.77 0.76 0.76 0.75;
        0.67 0.65 0.64 0.62;
        1.20 1.18 1.15 1.11;
        2.03 1.95 1.88 1.76;
        5.01 4.91 4.88 4.57;
        0.6 0.59 0.59 0.58;
        0.12 0.12 0.12 0.12;
        0.23 0.23 0.23 0.22;
        0.13 0.13 0.13 0.12];

gamma_net4 = [1.32 1.28 1.26 1.19;
        2.46 2.37 2.28 2.13;
        1.33 1.29 1.26 1.19;
        1.29 1.26 1.23 1.18;
        1.37 1.34 1.3 1.23;
        4.12 3.79 3.53 3.4;
        3.15 2.97 2.82 2.71;
        3.2 3.03 2.88 2.75;
        1.06 1.04 1.02 0.98;
        0.82 0.80 0.78 0.74;
        1.58 1.54 1.50 1.43;
        1.61 1.59 1.56 1.52;
        0.57 0.56 0.55 0.53;
        1.57 1.53 1.49 1.41;
        2.16 2.08 2.01 1.87;
        3.22 2.99 2.78 2.61;
        0.5 0.5 0.5 0.5;
        0.07 0.07 0.07 0.07;
        0.16 0.16 0.16 0.16;
        0.14 0.14 0.14 0.14];
    

orig_gamma = gamma_net1;
for idx = 1:20
    gamma_net1(idx, :) = gamma_net1(idx, :)./orig_gamma(idx, :);
    gamma_net2(idx, :) = gamma_net2(idx, :)./orig_gamma(idx, :);
    gamma_net3(idx, :) = gamma_net3(idx, :)./orig_gamma(idx, :);
    gamma_net4(idx, :) = gamma_net4(idx, :)./orig_gamma(idx, :);
end

% indices = [2 4 5 7 8 11 12 13 15 16 20];
% indices = [2 4 6 8 10 12 14 16 18 20];
indices = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];
P1 = zeros (length(indices), 1);
P2 = zeros (length(indices), 1);
P3 = zeros (length(indices), 1);
P4 = zeros (length(indices), 1);

for idx = 1:1:20
    elem = indices(idx);
    P1(idx) = mean(gamma_net1(elem, :));
    P2(idx) = mean(gamma_net2(elem, :));
    P3(idx) = mean(gamma_net3(elem, :));
    P4(idx) = mean(gamma_net4(elem, :));
end

P5 = gamma_net1(:, 1);
P6 = gamma_net2(:, 1);
P7 = gamma_net3(:, 1);
P8 = gamma_net4(:, 1);
bar1 = [P1 P2 P3 P4];
x = indices;
b = bar(x, bar1);
% b(1).DisplayName = 'Network-1';
% legendnames{1} = 'Network-1';
% legendnames{2} = 'Network-2';
% legendnames{3} = 'Network-3';
% legendnames{4} = 'Network-4';
% legend(b, legendnames);
leg1 = legend({'$N_{\Phi^{-1}}^1$', '$N_{\Phi^{-1}}^2$', '$N_{\Phi^{-1}}^3$', '$N_{\Phi^{-1}}^4$'},'Orientation','vertical', 'NumColumns', 2);
set (leg1, 'Interpreter', 'latex');
set (leg1, 'FontSize', 20);
xlabel('System', 'FontSize', 24, 'FontName', 'Serif', 'Interpreter','LaTex');
ylabel('$\gamma$', 'FontSize', 24, 'FontName', 'Serif', 'Interpreter','LaTex');
ylim([0 1.5]);
% set(gca,'xticklabel',{'1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'});
% somenames={'1'; '2'; '3'; '4'};
% set(gca, 'yticklabel', somenames);
% b(1).FaceColor = [.2 .6 .5];
% xtips1 = b(1).XEndPoints;
% ytips1 = b(1).YEndPoints;
% labels1 = string(b(1).YData);
% 
% text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
%     'VerticalAlignment','bottom')
% bar2 = [P1 P4];
% hold on
% plot(P5)
% plot (P6)
% plot(P7)
% plot (P8)
% hold off
