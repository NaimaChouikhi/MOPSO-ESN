function  z=ZDTLorBiobj(x)
[X Y Z] = lorenz(50, 10, 8/3);


% seed random generator
rand('state', sum(100*clock));

STEP = 6;

Lorenz=[X Y Z];

X=X';
E=X';
[dim1 dim2]=size(X);

B=max(X);

% 
% for i=1:dim1
%     for j=1:dim2
%         X(i,j)=X(i,j)/B;
%     end
% end
% A=min(X);
% 
% for i=1:dim1
%     for j=1:dim2
%         X(i,j)=X(i,j)-A;
%     end
% end
for i=1:dim1
    for j=1:dim2
        X(i,j)=X(i,j)/100;
    end
end


% X=tanh(Lorenz(1,:));
% Y=tanh(Lorenz(2,:));
% Z=tanh(Lorenz(3,:));
% unit counts (input, hidden, output)
IUC = 1;
%HUC = 100;
OUC = 1;

% initialize ESN weights
% 5% of recurrent weights set to 0.4 or -0.4
% no input weights
% 100% backward weights set randomly from [-1.00, 1.00]
%probInp  = [  1.00 ];
rngInp   = [  1.00 ]; 
%probRec  = [  0.15];
rngRec   = [ -0.6 ];
%probBack = [  0.1 ];
rngBack  = [ -0.1];
HUC=x(1);
probRec=x(2);
probBack=x(3);
probInp=x(4);
f1=x(1);
f2=x(2);
%rngRec=x(4);
%rngBack=x(5);
%rngInp=x(6);
%Lorenz(1,:)=tanh(Lorenz(1,:));
IP=X(1:500-STEP);
%IP=IP';
%[m1,n1]=size(IP)
 TP=X(1+STEP:500);
 IP1=X(501:1000-STEP);
 TP1=X(501+STEP:1000);
 IPT=X(1001:1500-STEP);
%IP=IP'
TPT=X(1001+STEP:1500);
%TP=TP';

Output=[];

  Lambda = 0.0;
UnitAct = 11;                            
                   % targets of the test database
% IP=outputSeq(1:2998)';
% plot(outputSeq);
% TP=outputSeq(3:3000)';
%  IP1=outputSeq(2999:3998)';
%  TP1=outputSeq(3001:4000)';
%  IPT=outputSeq(3999:end-2)';
% %IP=IP'
% TPT=outputSeq(4001:end)';
% create esn network
[net] = rnn_esn_new(IUC, HUC, OUC, probInp, rngInp, probRec, rngRec, probBack, rngBack, Lambda, UnitAct);
%fprintf('Maximal eig. (before scaling) was %f\n', max_eig);
%IP=trainInputSequence';
%IP=IP';
%[m1,n1]=size(IP);
%TP=trainOutputSequence';
% train network using all values, noise of 1e5 is added to states
[net, MSE, AO, ACT] = rnn_esn_train(net, IP, TP,200, 0.0);
%fprintf('Training MSE is %g\n', MSE);


% test network using 250 values of MG seq. as initial (teacher-forced) sequence
[AO, ACT] = rnn_esn_sim(net, IP1, TP1,0, 0.0);
MSE_Test = eval_mse(AO, TP1);
%fprintf('Testing MSE is %g\n', MSE)
% plot results
    % enregistrer les sortie du r�seau dans une matrice
f3=MSE_Test;
z=[f1
   f3];





