% maLRR--Multi-Center Adaptation Framework with Low-Rank Representation
%  The code requires a GPU to run
clc;clear all;close all;
load('data.mat')

Train_Target_Data = r';source_data{1,1} = r2';source_data{1,2} = r3';

[Z,~,~,W,~] = maLRR(Train_Target_Data,source_data,2,150,1e-3,1);

New_Train_Data = W * Train_Target_Data;New_Train_Data=New_Train_Data';
New_Source_r2 = W * Train_Target_Data * Z{1,1};New_Source_r2=New_Source_r2';
New_Source_r3 = W * Train_Target_Data * Z{1,2};New_Source_r3=New_Source_r3';

plot(New_Train_Data(:,1),New_Train_Data(:,2),'s','color',[79 129 189]/255,'Markerfacecolor',[79 129 189]/255,...
    'markersize',8);hold on
plot(New_Source_r2(:,1),New_Source_r2(:,2),'^','color',[192 80 77]/255,'Markerfacecolor',[192 80 77]/255,...
    'markersize',8);hold on
plot(New_Source_r3(:,1),New_Source_r3(:,2),'o','color',[155 187 89]/255,'Markerfacecolor',[155 187 89]/255,...
    'markersize',8)
set(gca,'FontSize',12,'FontName','Times New Roman');
xlabel({'x';'After adaptation'},'FontName','Times New Roman','FontSize',12);
ylabel('y', 'FontSize', 12,'FontName','Times New Roman');
legend({'Target','Source1','Source2'},'Location','Northwest','FontSize',12,'FontName','Times New Roman')