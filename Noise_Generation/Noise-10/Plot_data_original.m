function Plot_data_original(Xs,Xs_Label)

% 手动设置六种对比明显的颜色
colors = [
    1 0 0; % 红
    0 1 0; % 绿
    0 0 1; % 蓝
    0 1 1; % 青
    0.5 0 0.5; % 紫
    1 0.5 0; % 橙
    0 0.5 0; % 
    0 0.5 0.5; %
    1 1 0; %
    1 0 1; % 
];

[ns,~] = size(Xs);
% Set parameters, they are same as the paper "https://lvdmaaten.github.io/tsne/User_guide.pdf"
numDims = 2; 
pcaDims = 2; 
perplexity = 30; 
theta = .5; 
alg = 'svd';
% Run fast_tsne
mappedXs = fast_tsne(Xs, numDims, pcaDims, perplexity, theta, alg);
% Plot results
% 设置 colormap
cmap = colormap(gca, colors);
% cmap = hsv;
% interval = floor(64/length(unique(Xs_Label)));

interval = 1;

scatter(mappedXs(:,1),mappedXs(:,2),100,cmap(interval*Xs_Label,:),'filled');

% remove the numbers on the axis
set(gca,'XTick',[])
set(gca,'YTick',[])
box off
ax2 = axes('Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
box on