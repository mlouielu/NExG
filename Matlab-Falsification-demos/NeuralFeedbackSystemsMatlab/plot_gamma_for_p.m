
gamma = [1.87 1.81 1.76 1.66;
         1.32 1.28 1.26 1.19];

% for idx = 1:2
%     gamma(idx, :) = gamma(idx, :)./gamma(idx, 1);
% end

bar1 = [gamma(:, 1) gamma(:, 2) gamma(:, 3) gamma(:, 4)];
b = bar(bar1, 'FaceColor', 'flat');
for k = 1:size(bar1,2)
    b(k).CData = k;
end
% b(1).FaceColor = [.2 .6 .5];

leg1 = legend({'$~~p=1$', '$~~p=5$', '$~~p=10$', '$~~p=20$'}, 'Orientation','horizontal', 'NumColumns', 2, 'FontWeight', 'bold');
set (leg1, 'Interpreter', 'latex');
set (leg1, 'FontSize', 20);

xlabel('System 1', 'FontSize', 24, 'FontName', 'Serif', 'Interpreter','LaTex');
ylabel('$\gamma$', 'FontSize', 24, 'FontName', 'Serif', 'Interpreter','LaTex');
set(gca,'xticklabel',{'$N_{\Phi^{-1}}^1$','$N_{\Phi^{-1}}^4$'},'TickLabelInterpreter', 'latex')
% ylim([0 1.25]);