
clear all
clc

tic

 load dataset.mat; %Partition Dataset for the 5 fold test


% rng('default');     % reset random generator.
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance.  
opts.maxIter = 1000; % maximum iteration number of optimization.


for n=1:5
    test_VBM=Xt_VBM{1,n};
     test_CC=Xt_CC{1,n};
     test_Yt=Yt_SNP{1,n}(:,1); 

    task.DT{1}=X_VBM{1,n};
    task.DT{2}=X_CC{1,n};
    respons=Y_SNP{1,n}(:,1);
    task.target{1}=Y_SNP{1,n}(:,1); %1:rs429358
    task.target{2}=Y_SNP{1,n}(:,1); %2:rs429358
    task.lab{1}=Y{1,n};
    task.lab{2}=Y{1,n};
    gnd=task.lab{1};

    task.num=2;

 


 paraset=[0.0000001 0.0000003 0.000001 0.000003 0.00001];





    for j=1:length(paraset)
        opts.rho1=paraset(j);
        for k=1:length(paraset)
            opts.rho_L3=paraset(k);
            opts.init =2; % guess start point from data ZERO.
            kfold=5;
            kk=1;     
            % construct the index of cross_validation for each task.
            [tcv fcv]=f_myCV(gnd',kfold,kk); 
            %% begin to 5-fold.
            for cc=1:kfold 
                task.X = cell(task.num,1);
                task.Y = cell(task.num,1);
                for i=1:task.num                  
                    trLab=tcv{cc}';
                    % generate the task.              
                    task.X{i}=task.DT{i}(trLab,:);               
                    task.Y{i}=task.target{i}(trLab);
                    task.label{i}=task.lab{i}(trLab);
                end
                %----------Main Algorithm---------------        

                 
                  [S, D] = f_lapMatrix(task.Y{1});
                   [W, epsvalue] = f_MTM_APG(task.X,task.Y,opts,S, D);
                % find the selected features for each task. 
                trLab=tcv{cc}';
                teLab=fcv{cc}';
                pl=task.DT{1}(teLab,:)*W(:,1);
                pl2=task.DT{2}(teLab,:)*W(:,2);      

                et(cc)=sqrt(mean((pl-respons(teLab,1)).^2))+sqrt(mean((pl2-respons(teLab,1)).^2));                                    
                a=respons(teLab,1)-mean(respons(teLab,1));b=pl-mean(pl);
                a2=respons(teLab,1)-mean(respons(teLab,1));b2=pl2-mean(pl2);
              
                co(cc)=abs(sum(a.*b)/sqrt(sum(a.^2)*sum(b.^2)))+abs(sum(a2.*b2)/sqrt(sum(a2.^2)*sum(b2.^2)));%+abs(sum(a3.*b3)/sqrt(sum(a3.^2)*sum(b3.^2)))+abs(sum(a4.*b4)/sqrt(sum(a4.^2)*sum(b4.^2)));
            end       
            res_kfold_CO(kk)=mean(co);
            res_kfold_RMSE(kk)=mean(et);  
            res_CO(j,k)=mean(res_kfold_CO);
            res_RMSE(j,k)=mean(res_kfold_RMSE);
        end
    end
    ndim=size(res_RMSE);
    tempRMSE=10;
    tempCO=0;
    for ii=1:ndim(1)
        for jj=1:ndim(2)
               if  res_CO(ii,jj)>tempCO
                 tempCO=res_CO(ii,jj);
                 paraSet=[ii,jj];
               end
        end
    end
    paraSet

    test_opts.rho1=paraset(paraSet(1));
    test_opts.rho_L3=paraset(paraSet(2));
    Final_para{n}=paraSet;
    test_opts.tFlag = 1;     % terminate after relative objective value does not changes much.
    test_opts.tol = 10^-5;   % tolerance. 
    test_opts.maxIter = 1000; % maximum iteration number of optimization.
    test_opts.init = 2;  % guess start point from data ZERO.

        [newtask.S, newtask.D] = f_lapMatrix(task.DT{1});
        [newW, epsvalue] = f_MTM_APG(task.DT,task.target,opts,newtask.S, newtask.D);

    trainpl=task.DT{1}*newW(:,1);
    trainpl2=task.DT{2}*newW(:,2);

    trainRMSE(n)=sqrt(mean((trainpl-task.target{1}).^2));
    trainRMSE2(n)=sqrt(mean((trainpl2-task.target{2}).^2));

    aa=task.target{1}-mean(task.target{1});bb=trainpl-mean(trainpl);
    aa2=task.target{2}-mean(task.target{2});bb2=trainpl2-mean(trainpl2);

    trainCO(n)=sum(aa.*bb)/sqrt(sum(aa.^2)*sum(bb.^2));
    trainCO2(n)=sum(aa2.*bb2)/sqrt(sum(aa2.^2)*sum(bb2.^2));



    testpl=test_VBM*newW(:,1);  

     testpl2=test_CC*newW(:,2);

    testRMSE(n)=sqrt(mean((testpl-test_Yt).^2));
    testRMSE2(n)=sqrt(mean((testpl2-test_Yt).^2)); 


    Weight{n}=newW;

    aa=test_Yt-mean(test_Yt);bb=testpl-mean(testpl);
    aa2=test_Yt-mean(test_Yt);bb2=testpl2-mean(testpl2);

    testCO(n)=sum(aa.*bb)/sqrt(sum(aa.^2)*sum(bb.^2));
    testCO2(n)=sum(aa2.*bb2)/sqrt(sum(aa2.^2)*sum(bb2.^2));


    p{n}=testpl;
    p2{n}=testpl2;

end

toc
RMSE_VBM1=[mean( trainRMSE) std( trainRMSE)]
RMSE_CC2=[mean( trainRMSE2) std( trainRMSE2)]


CC_VBM1=[mean( trainCO) std( trainCO)]
CC_CC2=[mean( trainCO2) std( trainCO2)]


RMSE_VBM11=[mean(testRMSE) std(testRMSE)]
RMSE_CC22=[mean(testRMSE2) std(testRMSE2)]


CC_VBM11=[mean(testCO) std(testCO)]
CC_CC22=[mean(testCO2) std(testCO2)]


