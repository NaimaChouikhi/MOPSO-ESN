%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA121
% Project Title: Multi-Objective Particle Swarm Optimization (MOPSO)
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clear all;


%% Problem Definition

CostFunction=@(x) ZDTLor(x)     % Cost Function
nVar=4;             % Number of Decision Variables

VarSize=[1 nVar];   % Size of Decision Variables Matrix

VarMinSize=10;          % Lower Bound of Variables
VarMaxSize=100;          % Upper Bound of Variables
VarMinProbRec=0.005;          % Lower Bound of Variables
VarMaxProbRec=1;
VarMinProbBack=0.1;          % Lower Bound of Variables
VarMaxProbBack=1; 
VarMinProbInp=0.1;          % Lower Bound of Variables
VarMaxProbInp=1; % Upper Bound of Variables
VarMin=[VarMinSize VarMinProbRec VarMinProbBack VarMinProbInp ];
VarMax=[VarMaxSize VarMaxProbRec VarMaxProbBack VarMaxProbInp];


%% MOPSO Parameters

MaxIt=50;           % Maximum Number of Iterations

nPop=20;            % Population Size

nRep=20;            % Repository Size

w=0.5;              % Inertia Weight
wdamp=0.99;         % Intertia Weight Damping Rate
c1=0.1;               % Personal Learning Coefficient
c2=0.2;               % Global Learning Coefficient

nGrid=7;            % Number of Grids per Dimension
alpha=0.2;          % Inflation Rate

beta=3;             % Leader Selection Pressure
gamma=2;            % Deletion Selection Pressure

mu=0.5;             % Mutation Rate

%% Initialization

empty_particle.Position=[];
empty_particle.Velocity=[];
empty_particle.Cost=[];
empty_particle.Best.Position=[];
empty_particle.Best.Cost=[];
empty_particle.IsDominated=[];
empty_particle.GridIndex=[];
empty_particle.GridSubIndex=[];

pop=repmat(empty_particle,nPop,1);

for i=1:nPop
    
    pop(i).Position(1)=floor(unifrnd(VarMin(1),VarMax(1)));
    pop(i).Position(2)=unifrnd(VarMin(2),VarMax(2));
    pop(i).Position(3)=unifrnd(VarMin(3),VarMax(3));
    pop(i).Position(4)=unifrnd(VarMin(4),VarMax(4));
    pop(i).Velocity=zeros(VarSize);
    
    pop(i).Cost=CostFunction(pop(i).Position);
   
    
    % Update Personal Best
    pop(i).Best.Position=pop(i).Position;
    pop(i).Best.Cost=pop(i).Cost;
    
end
% Determine Domination
pop=DetermineDomination(pop);

rep=pop(~[pop.IsDominated]);

Grid=CreateGrid(rep,nGrid,alpha);

for i=1:numel(rep)
    rep(i)=FindGridIndex(rep(i),Grid);
end


%% MOPSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        leader=SelectLeader(rep,beta);
        
        pop(i).Velocity = w*pop(i).Velocity ...
            +c1*rand(VarSize).*(pop(i).Best.Position-pop(i).Position) ...
            +c2*rand(VarSize).*(leader.Position-pop(i).Position);
        
        pop(i).Position = pop(i).Position + pop(i).Velocity;
        pop(i).Position(1)=floor(pop(i).Position(1));
        
        pop(i).Position(1) = max(pop(i).Position(1), VarMin(1));
        pop(i).Position(1) = min(pop(i).Position(1), VarMax(1));
        for(compt=2:nVar)
         pop(i).Position(compt) = max(pop(i).Position(compt), VarMin(compt));
        pop(i).Position(compt) = min(pop(i).Position(compt), VarMax(compt));
        end
        pop(i).Cost = CostFunction(pop(i).Position);
        
        % Apply Mutation
        pm=(1-(it-1)/(MaxIt-1))^(1/mu);
        if rand<pm
            NewSol.Position=Mutate(pop(i).Position,pm,VarMin,VarMax);
            NewSol.Position(1)=floor(pop(i).Position(1));
            NewSol.Cost=CostFunction(NewSol.Position);
            if Dominates(NewSol,pop(i))
                pop(i).Position=NewSol.Position;
                pop(i).Cost=NewSol.Cost;

            elseif Dominates(pop(i),NewSol)
                % Do Nothing

            else
                if rand<0.5
                    pop(i).Position=NewSol.Position;
                    pop(i).Cost=NewSol.Cost;
                end
            end
        end
        pop(i).Position(1)=floor(pop(i).Position(1));
        pop(i).Position(1) = max(pop(i).Position(1), VarMin(1));
        pop(i).Position(1) = min(pop(i).Position(1), VarMax(1));
        for(compt2=2:nVar)
         pop(i).Position(compt2) = max(pop(i).Position(compt2), VarMin(compt2));
        pop(i).Position(compt2) = min(pop(i).Position(compt2), VarMax(compt2));
        end
        if Dominates(pop(i),pop(i).Best)
            pop(i).Best.Position=pop(i).Position;
            pop(i).Best.Cost=pop(i).Cost;
            
        elseif Dominates(pop(i).Best,pop(i))
            % Do Nothing
            
        else
            if rand<0.5
                pop(i).Best.Position=pop(i).Position;
                pop(i).Best.Cost=pop(i).Cost;
            end
        end
        
    end
    
    % Add Non-Dominated Particles to REPOSITORY
    rep=[rep
         pop(~[pop.IsDominated])]; %#ok
    
    % Determine Domination of New Resository Members
    rep=DetermineDomination(rep);
    
    % Keep only Non-Dminated Memebrs in the Repository
    rep=rep(~[rep.IsDominated]);
    
    % Update Grid
    Grid=CreateGrid(rep,nGrid,alpha);

    % Update Grid Indices
    for i=1:numel(rep)
        rep(i)=FindGridIndex(rep(i),Grid);
    end
    
    % Check if Repository is Full
    if numel(rep)>nRep
        
        Extra=numel(rep)-nRep;
        for e=1:Extra
            rep=DeleteOneRepMemebr(rep,gamma);
        end
        
    end
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Number of Repository Members = ' num2str(numel(rep))]);
    %pause;
    % Damping Inertia Weight
    w=w*wdamp;
%  pause  
end

%% Resluts
disp(' ');

EPC=[rep.Cost];
for j=1:(numel(rep))
    if (j==4)
        break;
    end
    disp(['Objective #' num2str(j) ':']);
    disp(['      Min = ' num2str(min(EPC(j,:)))]);
    disp(['      Max = ' num2str(max(EPC(j,:)))]);
    disp(['    Range = ' num2str(max(EPC(j,:))-min(EPC(j,:)))]);
    disp(['    St.D. = ' num2str(std(EPC(j,:)))]);
    disp(['     Mean = ' num2str(mean(EPC(j,:)))]);
    disp(' ');
    
end
   % Plot Costs
 figure(1);
 PlotCosts(pop,rep);
 title('Population (blue stars) convergence towards the Pareto Front (red stars)')
 xlabel('1^{st} Objective: Reservoir size ');
 ylabel('2^{nd} Objective: Reservoir connectivity rate');
 zlabel('3^{rd} Objective: Accuracy(RMSE) ');
saveas(gcf, '../results/fig5.png')
print('../results/plot', '-dpdf')