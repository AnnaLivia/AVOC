function [best_bound, best_Xopt, best_flag, best_ineq, cp_iter, cp_flag, ...
    avg_n_pair, avg_n_triangle, avg_n_clique, best_Bcell, best_l] = solve_cluster_cp_inherit(n_threads, ...
    WWt, Acell, b, k, original_n, original_trace, stopoption, maxiter, maxtime, ...
    tol, verbose, max_cp_iter, cp_tol, global_ub, bb_tol, ...
    eps_ineq, eps_active, max_ineq, max_pair_ineq, pair_perc, max_triangle_ineq, triangle_perc, ...
    parent_Bcell, parent_l, parent_n, inherit_perc, decision_i, decision_j)

% note that Z1 is the psd matrix and Z2 is the non-negative matrix.
addpath('/Library/gurobi1200/macos_universal2/matlab');

warning off;

%disp(' N_THREADS: ')
maxNumCompThreads(n_threads);
%disp(maxNumCompThreads)

n = size(WWt,1);

blk = cell(1);
blk{1, 1} = 's';
blk{1, 2} = n;

C = cell(1);
C{1} = -WWt;

At = svec(blk, Acell, 1);

L = cell(1);
L{1} = 0; % we want X >= 0

options.tol = tol;
options.printlevel = verbose;
options.stopoption = stopoption;
options.maxiter = maxiter;
options.maxtime = maxtime;

[Bcell, l] = inherit_cuts(parent_Bcell, parent_l, n, parent_n, inherit_perc, decision_i + 1, decision_j + 1);

Bt = svec(blk, Bcell, 1);

% solve with inherited inequalities
[~, X, ~, y, Z1, Z2, y2, ~, info, ~] = sdpnalplus(blk, At, C, b, L, [], Bt, l, [], options);
flag = info.termcode;

% sdp is not solved to the required accuracy
if flag == 1
    
    Bcell = cell(1, 0);
    l = zeros(0, 1);    
    Bt = svec(blk, Bcell, 1);
    
    % solve without inequalities
    [~, X, ~, y, Z1, Z2, ~, ~, info, ~] = admmplus(blk, At, C, b, L, [], [], [], [], options);
    flag = info.termcode;
    
    bound = safe_bound(blk, At, C, b, y, Z2, [], 0, 0, X);
    
else

    bound = safe_bound(blk, At, C, b, y, Z2, Bt, y2, l, X);
    
end

% cutting plane iterations
[best_bound, best_Xopt, best_flag, best_ineq, cp_iter, cp_flag, ...
    avg_n_pair, avg_n_triangle, avg_n_clique, best_Bcell, best_l] = cutting_plane_loop(...
    n_threads, blk, C, At, b, L, Z1, Z2, Bt, flag, bound, X, y, Bcell, l, ...
    k, original_n, original_trace, options, max_cp_iter, cp_tol, ...
    global_ub, bb_tol, eps_ineq, eps_active, max_ineq, max_pair_ineq, pair_perc, max_triangle_ineq, triangle_perc);

clear C At Bcell Bt l X y Z1 Z2

end
