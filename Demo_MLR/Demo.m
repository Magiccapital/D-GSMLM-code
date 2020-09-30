
clc
clear

Data = load('demoData.mat');
net = Data.demoData.Net;
label = Data.demoData.Label;

ordinalPatterns = fop_mining(net,label);

clc;