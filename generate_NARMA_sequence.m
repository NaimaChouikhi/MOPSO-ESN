function [inputSequence, outputSequence,outputSeq] = generate_NARMA_sequence(sequenceLength, memoryLength)
%  Generates a sequence using a nonlinear autoregressive moving average
% (NARMA) model. The sequence at the beginning includes a  ramp-up
% transient, which should be deleted if necessary. The NARMA equation to be
% used must be hand-coded into this function (at bottom)

% inputs: 
% sequenceLength: a natural number, indicating the length of the
% sequence to be generated
% memoryLength: a natural number indicating the dependency length
%
% outputs: 
% InputSequence: array of size sequenceLength x 2. First column contains 
%                uniform noise in [0,1] range, second column contains bias 
%                input (all 1's)             
% OutputSequence: array of size sequenceLength x 1 with the NARMA output
%
% usage example:
% [a b] = generate_linear_sequence(1000,10) ; 
%
% Created April 30, 2006, D. Popovici
% Copyright: Fraunhofer IAIS 2006 / Patent pending
% Revision H. Jaeger Feb 23, 2007

%%%% create input 
inputSequence = [rand(sequenceLength,1)*0.4];
outputSeq=zeros(sequenceLength,1);
outputSeq(1,1)=rand;
outputSeq(2,1)=rand;
for j = memoryLength + 1 : sequenceLength
    outputSeq(j,1)=1-0.1 * outputSeq(j-1,1)*outputSeq(j-1,1)+0.3*outputSeq(j-2,1);
end
outputSeq=tanh(outputSeq);
% use the input sequence to drive a NARMA equation

out=0;
outputSequence = 0.1*ones(sequenceLength,1); 
for j=1:memoryLength
    out=out+outputSequence(j,1);
end

for i = memoryLength + 1 : sequenceLength
    % insert suitable NARMA equation on r.h.s., this is just an ad hoc
    % example
    outputSequence(i,1) = 0.3 * outputSequence(i-1,1) + 0.05*outputSequence(i-1,1)*out +0.75.*(inputSequence(i-1,1))* inputSequence(i-memoryLength,1)+0.1;
end
outputSequence=tanh(outputSequence);
