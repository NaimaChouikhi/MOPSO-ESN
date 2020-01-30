function  z=Narma(x)
% seed random generator
rand('state', sum(100*clock));
setpath;

% unit counts (input, hidden, output)
IUC = 1;
%HUC = 50;
OUC = 1;

% initialize ESN weights
% 5% of recurrent weights set to 0.4 or -0.4
% no input weights
% 100% backward weights set randomly from [-1.00, 1.00]
%probInp  = [  1.00 ];
rngInp   = [  1.00 ]; 
%probRec  = [  0.05 ];
rngRec   = [ -0.6 ];
%probBack = [  0.1 ];
rngBack  = [ -0.1];
HUC=x(1);
probRec=x(2);
probBack=x(3);
probInp=x(4);
f1=x(1);
f2=x(2);
% create input and output time series
% IP = ones(0,500);
% TP = 0.5 .* sin((1:500)/5); 

sequenceLength = 5000;

%disp('Generating data ............');
%disp(sprintf('Sequence Length %g', sequenceLength ));

systemOrder = 10 ; % set the order of the NARMA equation
[inputSequence outputSequence outputSeq] = generate_NARMA_sequence(sequenceLength , systemOrder) ; 


%%%% split the data into train and test

% train_fraction = 0.75 ; % use 50% in training and 50% in testing
% [trainInputSequence, testInputSequence] = ...
%     split_train_test(inputSequence,train_fraction);
% [trainOutputSequence,testOutputSequence] = ...
%     split_train_test(outputSequence,train_fraction);

%[m2,n2]=size(TP);
%TP=TP';
IP=inputSequence(1:2000)';
%IP=IP';
%[m1,n1]=size(IP)
TP=outputSequence(1:2000)';
 IP1=inputSequence(2001:3000)';
 TP1=outputSequence(2001:3000)';
 IPT=inputSequence(3001:3500)';
%IP=IP'
TPT=outputSequence(3001:3500)';
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
[net] = rnn_esn_new(IUC, HUC, OUC, probInp, rngInp, probRec, rngRec, probBack, rngBack, Lambda, UnitAct);%[net, max_eig] = rnn_esn_new_narma_clerck( IP, TP, IP1, TP1,IPT, TPT, IUC, HUC, OUC, probInp, rngInp, probRec, rngRec, probBack, rngBack,Lambda,UnitAct);
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
%fprintf('Testing MSE is after PSO-pretraining %g\n', MSE)
% plot results
    % enregistrer les sortie du réseau dans une matrice
f3=MSE_Test;
z=[f1
   f2
   f3];





