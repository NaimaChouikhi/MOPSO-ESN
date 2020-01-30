%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA102
% Project Title: Implementation of Particle Swarm Optimization in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clear all;


%% Problem Definition

CostFunction=@(x) ZDTLorpso(x);        % Cost Function

nVar=4;            % Number of Decision Variables

VarSize=[1 nVar];   % Size of Decision Variables Matrix
VarMinSize=10;
VarMaxSize=100;          % Upper Bound of Variables
VarMinProbRec=0.005;          % Lower Bound of Variables
VarMaxProbRec=1;
VarMinProbBack=0.1;          % Lower Bound of Variables
VarMaxProbBack=1; 
VarMinProbInp=0.5;          % Lower Bound of Variables
VarMaxProbInp=1; % Upper Bound of Variables
VarMin=[VarMinSize VarMinProbRec VarMinProbBack VarMinProbInp ];
VarMax=[VarMaxSize VarMaxProbRec VarMaxProbBack VarMaxProbInp];


%% PSO Parameters

MaxIt=50;      % Maximum Number of Iterations

nPop=20;        % Population Size (Swarm Size)

% PSO Parameters
w=0.5;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=0.1;         % Personal Learning Coefficient
c2=0.2;         % Global Learning Coefficient

% If you would like to use Constriction Coefficients for PSO,
% uncomment the following block and comment the above set of parameters.

% % Constriction Coefficients
  phi1=2.05;
 phi2=2.05;
 phi=phi1+phi2;
 chi=2/(phi-2+sqrt(phi^2-4*phi));
 w=chi;          % Inertia Weight
 wdamp=1;        % Inertia Weight Damping Ratio
 c1=chi*phi1;    % Personal Learning Coefficient
 c2=chi*phi2;    % Global Learning Coefficient

% Velocity Limits
VelMax=0.1*(VarMax-VarMin);
VelMin=-VelMax;

%% Initialization

empty_particle.Position=[];
empty_particle.Cost=[];
empty_particle.Velocity=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];

particle=repmat(empty_particle,nPop,1);

GlobalBest.Cost=inf;

for i=1:nPop
    
    % Initialize Position
    %particle(i).Position=unifrnd(VarMin,VarMax,VarSize);
    particle(i).Position(1)=floor(unifrnd(VarMin(1),VarMax(1)));
    particle(i).Position(2)=unifrnd(VarMin(2),VarMax(2));
    particle(i).Position(3)=unifrnd(VarMin(3),VarMax(3));
    particle(i).Position(4)=unifrnd(VarMin(4),VarMax(4));
    
    % Initialize Velocity
    particle(i).Velocity=zeros(VarSize);
    
    % Evaluation
    particle(i).Cost=CostFunction(particle(i).Position);
    
    % Update Personal Best
    particle(i).Best.Position=particle(i).Position;
    particle(i).Best.Cost=particle(i).Cost;
    
    % Update Global Best
    if particle(i).Best.Cost<GlobalBest.Cost
        
        GlobalBest=particle(i).Best;
        
    end
    
end

BestCost=zeros(MaxIt,1);

%% PSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
        
        % Apply Velocity Limits
        
        %particle(i).Velocity = max(particle(i).Velocity,VelMin);
        %particle(i).Velocity = min(particle(i).Velocity,VelMax);
        for(compt=1:nVar)
         particle(i).Velocity(compt) = max(particle(i).Velocity(compt), VelMin(compt));
        particle(i).Velocity(compt) = min(particle(i).Velocity(compt), VelMax(compt));
        end
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Velocity Mirror Effect
       % IsOutside=(particle(i).Position<VarMin | particle(i).Position>VarMax);
       % particle(i).Velocity(IsOutside)=-particle(i).Velocity(IsOutside);
        
        % Apply Position Limits
        particle(i).Position = max(particle(i).Position,VarMin);
        particle(i).Position = min(particle(i).Position,VarMax);
         particle(i).Position(1)=floor(particle(i).Position(1));
        for(compt=1:nVar)
         particle(i).Position(compt) = max(particle(i).Position(compt), VarMin(compt));
        particle(i).Position(compt) = min(particle(i).Position(compt), VarMax(compt));
        end
        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost<particle(i).Best.Cost
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Cost=particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost<GlobalBest.Cost
                
                GlobalBest=particle(i).Best;
                
            end
            
        end
        
    end
    
    BestCost(it)=GlobalBest.Cost;
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    
end

BestSol = GlobalBest;

%% Results


figure(3);
semilogy(BestCost,'LineWidth',2);
title('best RMSE evolution')
xlabel('Iteration');
ylabel('Best Cost');
saveas(gcf, '../results/fig.png')
print('../results/plot', '-dpdf')