function [neto, oldSpecRad] = rnn_esn_new(IUC, HUC, OUC, probInp, rngInp, probRec, rngRec, probBack, rngBack, specRad, unitAct)
% [neto, oldSpecRad] = rnn_esn_new(IUC, HUC, OUC, probInp, rngInp, probRec, rngRec, probBack, rngBack, specRad, unitAct)
% neto - new network structure
% oldSpecRad - old spectral radius of recurrent weight matrix
% IUC - number of input units
% HUC - number of hidden units
% OUC - number of output units
% probInp  - vector of prob. of input weights
% rngInp   - vector of input weights' ranges
% probRec  - vector of prob. of recurrent weights
% rngRec   - vector of recurrent weights' ranges
% probBack - vector of prob. of backward weights
% rngBack  - vector of backward weights' ranges
% specRad  - required spectral radius (if omited or zero -> no scaling)
% unitAct  - units' activation function specifier (0 - tanh, 1 - lin. for hid. units

% set number of all units
AUC = 1+IUC+HUC+OUC;

% set numbers of units
net.numInputUnits    = IUC;
net.numHiddenUnits   = HUC;
net.numOutputUnits   = OUC;
net.numAllUnits      = AUC;

% set neuron masks
net.maskInputUnits   = [0; ones(IUC, 1); zeros(AUC-1-IUC, 1)];
net.maskOutputUnits  = [zeros(AUC-OUC, 1); ones(OUC, 1)];
net.indexOutputUnits = find(net.maskOutputUnits);
net.indexInputUnits  = find(net.maskInputUnits);


% set weight matrices
inputWeights = zeros(HUC, IUC+1, length(probInp));
recurrentWeights = zeros(HUC, HUC, length(probRec));
backwardWeights = zeros(HUC, OUC, length(probBack));

for d=(1:length(probInp))
    inputWeights(:,:,d) = init_weights(inputWeights(:,:,d), probInp(d),rngInp(d));
end;

for d=(1:length(probRec))
    recurrentWeights(:,:,d) = init_weights(recurrentWeights(:,:,d), probRec(d),rngRec(d));
end;

for d=(1:length(probBack))
    backwardWeights(:,:,d) = init_weights(backwardWeights(:,:,d), probBack(d),rngBack(d));
end;

    
% init parameters
if nargin<10, specRad = 0.0; end;


% scale to defined spectral radius
oldSpecRad = NaN;
if (nargout>1 || specRad>0) && length(probRec)>=1,
    oldSpecRad = max(abs(eig(recurrentWeights(:,:,1))));
end;

if specRad>0,
    recurrentWeights = recurrentWeights ./ oldSpecRad .* specRad;
end;

% init parameters
if nargin<11, unitAct = 0; end;

% set units
unit = struct('actFunc',1,'actFuncC1',0.0,'actFuncC2',1.0);
for i=(1:AUC), net.units(i) = unit; end;
if unitAct==21, for i=(IUC+2:AUC-OUC), net.units(i).actFunc = 2; end; 
elseif unitAct==22; for i=(IUC+2:AUC), net.units(i).actFunc = 2; end;
elseif unitAct==12; for i=(AUC-OUC+1:AUC), net.units(i).actFunc = 2; end;
elseif unitAct==0;
elseif unitAct==11;
else error('Unknown unit activation function specifier.'); 
end;

% set weights
weight = struct('dest',0,'source',0,'delay',0,'value',0);

nw = 1;
for i=(IUC+2:AUC-OUC),
    % input weights
    for j=(1:IUC+1),
        for d=(1:length(probInp)),
            value = inputWeights(i-IUC-1, j, d);%init_weight(probInp(d),rngInp(d));
            if value ~= 0,
                net.weights(nw).value  = value;
                net.weights(nw).dest   = i;
                net.weights(nw).source = j;
                net.weights(nw).delay  = d-1;
                nw = nw+1;
            end;
        end;
    end;
    
    % recurrent weights
    for j=(IUC+2:AUC-OUC),
        for d=(1:length(probRec)),
            value = recurrentWeights(i-IUC-1, j-IUC-1, d);
            if value ~= 0,
                net.weights(nw).value  = value;
                net.weights(nw).dest   = i;
                net.weights(nw).source = j;
                net.weights(nw).delay  = d;
                nw = nw+1;
            end;
        end;
    end;
    
    % backward weights
    for j=(AUC-OUC+1:AUC),
        for d=(1:length(probBack)),
            value = backwardWeights(i-IUC-1, j-AUC+OUC);
            if value ~= 0,
                net.weights(nw).value  = value;
                net.weights(nw).dest   = i;
                net.weights(nw).source = j;
                net.weights(nw).delay  = d;
                nw = nw+1;
            end;
        end;
    end;
end;


% set number of weights
net.numWeights = nw-1;
net.firstForwardWeight = nw;
    
% initialize starting activities from [0, 1]
net.maxDelay = max([length(probInp)-1, length(probRec), length(probBack)]);

% initialize initial activations from [-1, 1]
net.actInit = zeros(net.numAllUnits, net.maxDelay);
% net.actInit = 2.0 * rand(net.numAllUnits, net.maxDelay) - 1.0;

neto = net;



% initialze weight 
function weight = init_weight(prob, rng)

weight = 0;
if rand > prob, return; end;

weight = 2.0 * rand - 1.0;
if rng >= 0, 
    weight = weight .* rng;
else 
    if weight  < 0; weight =  rng;
    else weight = -rng; 
    end;
end; 



% initialze weights given as inputs
function weights = init_weights(weights, prob, rng)

mask = rand(size(weights)) < prob;
weights = (2.0 * rand(size(weights)) - 1.0) .* mask;
if rng >= 0, 
    weights = weights .* rng;
else 
    weights(weights < 0) = rng; 
    weights(weights > 0) = -rng;
end; 
