clc
clear all

load('Simulation.mat')
num=5;
lambda=0.01;
gamma=0.005;
nu=10^(-5);

[W,Omega]=MTERL(data,label,lambda,gamma,nu);

clear num lambda gamma nu



