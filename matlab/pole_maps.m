function pole_maps(outDir)
% POLE_MAPS  3-D pole maps of the Mellin-Barnes kernels in Appendix C (Calculation Discrepancies).
%
%   pole_maps            % writes the seven PNGs into ../figs
%   pole_maps('out')     % writes them into ./out
%
% Each panel plots  H(x,y) = log(1 + |Gamma(z)/(shift + z)|),  z = x + iy,  where
%     I(s)  kernel:  Gamma(z)/(eps + z)        (extra pole at z = -eps)
%     Q2(s) kernel:  Gamma(z)/(eps - 1 + z)    (extra pole at z = 1 - eps)
% The entire factor s^{-z} is dropped: it has no zeros or poles, so no pole moves, and dropping it
% keeps all poles at comparable height. Evaluation points are nudged half a grid cell off-lattice so
% no point lands on a pole; nothing is masked.
%
%   File (report label)                                     kernel  eps values
%   mellin_barnes_pole_collision        (fig:discrep-mb-overview)  I     0.35, 0, -0.35
%   plot_2_I_positive_epsilon_poles     (fig:discrep-I-pos)        I     0.35
%   plot_3_I_negative_epsilon_poles     (fig:discrep-I-neg)        I     -0.35
%   plot_4_I_epsilon_collision_triptych (fig:discrep-I-collision)  I     -0.75, 0.25, 0
%   plot_5_Q2_epsilon_below_one         (fig:discrep-Q2-below)     Q2    0.25
%   plot_6_Q2_epsilon_between_one_and_two (fig:discrep-Q2-above)   Q2    1.5
%   plot_7_Q2_epsilon_one_collision     (fig:discrep-Q2-collision) Q2    1
%
% Base MATLAB only (R2019b+ for tiledlayout/exportgraphics); also runs in GNU Octave for previews.
% MATLAB's gamma() is real-only, so a complex Lanczos Gamma is included below.

if nargin < 1, outDir = fullfile(fileparts(mfilename('fullpath')), '..', 'figs'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
close all;

specs = {
    'mellin_barnes_pole_collision',          'I',  [0.35 0 -0.35]
    'plot_2_I_positive_epsilon_poles',       'I',  0.35
    'plot_3_I_negative_epsilon_poles',       'I',  -0.35
    'plot_4_I_epsilon_collision_triptych',   'I',  [-0.75 0.25 0]
    'plot_5_Q2_epsilon_below_one',           'Q2', 0.25
    'plot_6_Q2_epsilon_between_one_and_two', 'Q2', 1.5
    'plot_7_Q2_epsilon_one_collision',       'Q2', 1
};

% Grid over the complex plane (moderate resolution keeps the mesh facets visible)
x = linspace(-3, 1, 121);  y = linspace(-2, 2, 121);
[X, Y] = meshgrid(x, y);
cmap = local_pole_colormap(256);

for i = 1:size(specs, 1)
    [name, kern, epsv] = specs{i, :};
    make_figure(fullfile(outDir, [name '.png']), kern, epsv, X, Y, cmap);
end
end


%% ======================================================================== figure
function make_figure(file, kern, epsv, X, Y, cmap)
n = numel(epsv);
H = cell(1, n);  ztop = zeros(1, n);
for k = 1:n
    shift = epsv(k) - strcmp(kern, 'Q2');          % I: eps,  Q2: eps - 1
    H{k} = local_compute_panel(X, Y, shift, 20);
    ztop(k) = ceil(max(H{k}(:)) * 10) / 10 + 0.3;
end
% simple-pole panels share one z-range; a double-pole panel (shift = 0) is genuinely taller
dbl = abs(epsv - strcmp(kern, 'Q2')) < 1e-12;
if any(~dbl), ztop(~dbl) = max(ztop(~dbl)); end

if strcmp(kern, 'I')
    zlab = '$\log\!\left(1+\left|\Gamma(z)/(\varepsilon+z)\right|\right)$';
    head = '$|\Gamma(z)/(\varepsilon+z)|$';
else
    zlab = '$\log\!\left(1+\left|\Gamma(z)/(\varepsilon-1+z)\right|\right)$';
    head = '$|\Gamma(z)/(\varepsilon-1+z)|$';
end

if n == 1, pos = [60 60 820 680]; else, pos = [60 60 1900 680]; end
fig = figure('Color', 'k', 'Position', pos);
useTiles = exist('tiledlayout', 'file') || exist('tiledlayout', 'builtin');
if useTiles, t = tiledlayout(fig, 1, n, 'TileSpacing', 'compact', 'Padding', 'compact'); end

for k = 1:n
    if useTiles, ax = nexttile(t, k); else, ax = subplot(1, n, k); end
    ttl = sprintf('$\\varepsilon=%g$', epsv(k));
    if dbl(k), ttl = [ttl ' (double pole)']; end
    local_render_panel(ax, X, Y, H{k}, [-3 1], [-2 2], [0 ztop(k)], [0 ztop(k)], -37, 22, cmap, ttl, zlab);
end

if n > 1 && exist('sgtitle', 'file')
    sgtitle(fig, [head ' over the complex $z$-plane (plotted as $\log(1+|\cdot|)$ to keep the poles finite)'], ...
        'Interpreter', 'latex', 'Color', [0.92 0.92 0.96], 'FontSize', 13);
end

set(fig, 'InvertHardCopy', 'off');                 % keep the black background on export
if exist('exportgraphics', 'file')
    exportgraphics(fig, file, 'Resolution', 300, 'BackgroundColor', 'k');
else
    print(fig, file, '-dpng', '-r300');
end
fprintf('Saved %s\n', file);
close(fig);
end


%% ======================================================================== helpers
function H = local_compute_panel(X, Y, shift, H_cap)
% H = log(1 + |Gamma(z)/(shift + z)|) on the full grid, nudged half a cell off-lattice.
dx = X(1,2) - X(1,1);  dy = Y(2,1) - Y(1,1);
Z = (X + dx/2) + 1i*(Y + dy/2);
F = local_complex_gamma(Z) ./ (shift + Z);
F(~isfinite(F)) = H_cap;                          % safety net only
H = min(log(1 + abs(F)), H_cap);
end

function g = local_complex_gamma(z)
% Lanczos approximation (g = 7, n = 9) with reflection for Re(z) < 1/2.
p = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, ...
     -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, ...
     1.5056327351493116e-7];
z = complex(z);
refl = real(z) < 0.5;
zz = z;  zz(refl) = 1 - z(refl);  zz = zz - 1;
x = p(1) * ones(size(zz));
for i = 1:8, x = x + p(i+1) ./ (zz + i); end
t = zz + 7.5;
y = sqrt(2*pi) .* t.^(zz + 0.5) .* exp(-t) .* x;
g = y;
g(refl) = pi ./ (sin(pi * z(refl)) .* y(refl));
end

function cmap = local_pole_colormap(n)
% light blue -> pink -> magenta -> deep purple (cool valley floor, hot spike tips)
stops = [0.00 0.72 0.86 0.97
         0.30 0.95 0.74 0.84
         0.55 0.86 0.42 0.74
         0.80 0.55 0.22 0.72
         1.00 0.18 0.10 0.46];
tt = linspace(0, 1, n)';
cmap = zeros(n, 3);
for c = 1:3, cmap(:, c) = interp1(stops(:,1), stops(:,c+1), tt, 'pchip'); end
cmap = min(max(cmap, 0), 1);
end

function local_render_panel(ax, X, Y, H, x_lim, y_lim, z_lim, c_lim, az, el, cmap, ttl, zlab)
axes(ax); hold(ax, 'on');
surf(ax, X, Y, H, 'FaceColor', 'interp', 'EdgeColor', [0.06 0.06 0.06], 'LineWidth', 0.35, 'FaceAlpha', 1);
colormap(ax, cmap);  caxis(ax, c_lim);
xlim(ax, x_lim); ylim(ax, y_lim); zlim(ax, z_lim);
view(ax, az, el);
box(ax, 'on'); grid(ax, 'on');
axcol = [0.82 0.82 0.88];
set(ax, 'Color', 'k', 'XColor', axcol, 'YColor', axcol, 'ZColor', axcol, ...
    'GridColor', [0.45 0.45 0.5], 'GridAlpha', 0.35, 'FontSize', 10);
xlabel(ax, '$\mathrm{Re}(z)$', 'Interpreter', 'latex', 'Color', axcol, 'FontSize', 12);
ylabel(ax, '$\mathrm{Im}(z)$', 'Interpreter', 'latex', 'Color', axcol, 'FontSize', 12);
zlabel(ax, zlab, 'Interpreter', 'latex', 'Color', axcol, 'FontSize', 11);
title(ax, ttl, 'Interpreter', 'latex', 'Color', [0.95 0.95 1], 'FontSize', 13);
camlight(ax, 'headlight');  lighting(ax, 'gouraud');  material(ax, [0.55 0.85 0.25 8]);
hold(ax, 'off');
end
