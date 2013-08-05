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

% plot the frequency response of the filter
file = fopen("high_pass1.view", "r");
[mag_H, N] = fread(file, Inf, "single");
fclose(file);
n = (1:N/2)';
fs = 2 * pi;  % units in radians / sample
ns = n .* fs ./ N;
figure();
plot(ns, mag_H(n));

title('High-pass Filter Response');
xlabel('angular frequency (radians/sample)');
ylabel('magnitude |H(n)|'); 
print("high_pass1.svg", "-dsvg");

% plot the frequency response of the signal before filtering
file = fopen("high_pass2.view", "r");
mag_Y = fread(file, Inf, "single");
fclose(file);
figure();
plot(ns, mag_Y(n), "1");
axis([0 pi 0 220]);

title('Frequency Spectrum of Input Signal x(n) (Before High-Pass Filter)');
xlabel('angular frequency (radians/sample)');
ylabel('magnitude |X(n)|'); 
print("high_pass2.svg", "-dsvg");

% plot the frequency response of the signal after filtering
file = fopen("high_pass3.view", "r");
mag_Y = fread(file, Inf, "single");
fclose(file);
figure();
plot(ns, mag_Y(n), "3")
axis([0 pi 0 220]);

title('Frequency Spectrum of Output Signal y(n) (After High-Pass Filter)');
xlabel('angular frequency (radians/sample)');
ylabel('magnitude |Y(n)|'); 
print("high_pass3.svg", "-dsvg");

