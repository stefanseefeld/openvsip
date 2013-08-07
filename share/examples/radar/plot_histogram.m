%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (c) 2010, 2011 CodeSourcery, Inc.  All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above
%       copyright notice, this list of conditions and the following
%       disclaimer in the documentation and/or other materials
%       provided with the distribution.
%     * Neither the name of CodeSourcery nor the names of its
%       contributors may be used to endorse or promote products
%       derived from this software without specific prior written
%       permission.
%
% THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
% EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
% BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
% OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
% EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Plot the histogram created in the SV++ clutter example.  After running
%%%  this script the image will be saved in 'histogram.svg' for later viewing.

%% Input values, make sure these match the values specified on lines 145, 146,
%%  and 156.
min_value = 0;
max_value = 5;
file_name = 'histogram.view';

fid = fopen(file_name, 'rb');

data = fread(fid, 'real*4');
bins = linspace(min_value, max_value, length(data));
plot(bins, data, 'bo', 'LineWidth', 3);
grid on
xlabel('Value')
ylabel('Frequency')
title('Weibull Distribution')
legend('\lambda = 1.0, k = 1.5')

print -dsvg histogram.svg;

fclose(fid);
