#include <armadillo>
#include <iomanip>
#include <random>
#include "mount_model.h"
#include "sdp_branch_and_bound.h"
#include "ThreadPoolPartition.h"
#include "ac_heuristic.h"
#include "config_params.h"

// Save data to file
void save_to_file(arma::mat& X, std::string name)
{
	std::ofstream f(result_path + "_" + name + ".txt");
	if (!f)
	{
		std::cerr << "save_to_file: cannot open\n";
		return;
	}
	f.setf(std::ios::fmtflags(0), std::ios::floatfield);
	f << std::setprecision(17);
	for (arma::uword i = 0; i < X.n_rows; ++i)
	{
		f << X(i, 0);
		for (arma::uword j = 1; j < X.n_cols; ++j) f << ' ' << X(i, j);
		f << '\n';
	}
}

// Read partial clustering sol
arma::mat read_part_sol(const char* filename, int nh)
{
	std::ifstream file(filename);
	if (!file)
	{
		std::cerr << strerror(errno) << "\n";
		exit(EXIT_FAILURE);
	}

	// read the values according to the number of columns
	arma::mat sol(nh, k);
	for (int i = 0; i < nh; i++)
	{
		sol.row(i) = arma::zeros(k).t();
		double c = 0;
		file >> c;
		sol(i, c) = 1;
	}
	return sol;
}

// Read partial clustering value
double read_part_val(const char* filename)
{
	std::ifstream file(filename);
	if (!file) {
		std::cerr << std::strerror(errno) << "\n";
		std::exit(EXIT_FAILURE);
	}
	double value{};
	if (!(file >> value)) {
		std::cerr << "Failed to read value from: " << filename << "\n";
		std::exit(EXIT_FAILURE);
	}
	return value;
}

// Evaluate a partition using Kmeans with provided centroids_heu
std::pair<double,double>
evaluate_partition_idx(const arma::mat& data,
                       const std::vector<std::vector<int>>& init_sol,
                       arma::mat& centroids_heu) // k x d
{
    double obj_sum = 0.0;
    double cdist_sum = 0.0;

    for (int h = 0; h < p; ++h)
    {
        const auto& idx_list = init_sol[h];
        if (idx_list.empty()) continue;

        arma::uvec idx(idx_list.size());
        for (size_t t = 0; t < idx_list.size(); ++t) idx(t) = static_cast<arma::uword>(idx_list[t]);

        arma::mat data_antic = data.rows(idx);

    	// Run KMeans initialized at 'centroids_heu'
    	Kmeans km(data_antic, k, kmeans_verbose);
    	km.start(kmeans_max_it, 1, centroids_heu);

        obj_sum += km.objectiveFunction();

        // centroid distance
		arma::mat new_cent = km.getCentroids();
    	cdist_sum += arma::accu(arma::sqrt(arma::sum(arma::square(new_cent - centroids_heu), 1)));
    }

    return {obj_sum, cdist_sum};
}

// Create first solution for the anticlustering heuristic
arma::mat create_first_sol(arma::mat& data,
						   arma::mat& antic_sol,
						   arma::mat& centroids_heu,
						   arma::mat& ub_sol,
						   std::vector<std::vector<std::vector<int>>>& points,
						   std::vector<std::vector<int>>& sizes)
{
	arma::mat obj(p, k, arma::fill::zeros);

	for (int h = 0; h < p; ++h)
	{
		arma::uvec idx = arma::find(antic_sol.col(h) == 1.0);
		const arma::uword nh = idx.n_elem;
		arma::mat data_antic = data.rows(idx);

		// call python to compute first k-means with improved heuristic
		save_to_file(data_antic, "pi_" + std::to_string(h));
		std::string in_path = result_path + "_pi_" + std::to_string(h) + ".txt";
		std::string out_path = result_path + "_in_" + std::to_string(h) + ".txt";
		std::string args = in_path + " " + std::to_string(k) + " " + out_path;
		std::string command = "python3 ./run_kmeans.py " + args + " 0";
		std::cout << command << "\n";
		int system_val = std::system(command.c_str());
		if (system_val == -1)
		{
			std::cout << "Failed to call Python script\n";
			std::exit(EXIT_FAILURE);
		}

		// read assignments
		arma::mat sol = read_part_sol(out_path.c_str(), static_cast<int>(nh));

		// centroids and per-cluster obj
		arma::vec counts = arma::sum(sol, 0).t();
		if (arma::any(counts == 0))
		{
			std::cout << "create_first_sol(): empty cluster in partition %d!\n";
			std::exit(EXIT_FAILURE);
		}
		arma::mat centroids = sol.t() * data_antic;
		centroids.each_col() /= counts;
		arma::mat resid = data_antic - sol * centroids;
		arma::vec row_sse = arma::sum(arma::square(resid), 1);
		arma::vec obj_h(k, arma::fill::zeros);
		for (int c = 0; c < k; ++c)
			obj_h(c) = arma::dot(row_sse, sol.col(c));

		// --- centroid mapping to heuristic centroids
		// D2 = ||centroids||^2 + ||centroids_heu||^2 - 2*centroids*centroids_heu^T
		arma::vec n1 = arma::sum(arma::square(centroids), 1);
		arma::vec n2 = arma::sum(arma::square(centroids_heu), 1);
		arma::mat D1 = n1 * arma::ones<arma::rowvec>(k)
			+ arma::ones<arma::vec>(k) * n2.t()
			- 2.0 * (centroids * centroids_heu.t()); // k x k

		// for each found cluster c, pick the best unused heuristic cluster
		std::vector<int> mapping(k, -1);
		std::vector<char> used(k, 0);
		for (int c1 = 0; c1 < k; ++c1)
		{
			double best = std::numeric_limits<double>::infinity();
			int best_c2 = -1;
			for (int c2 = 0; c2 < k; ++c2)
			{
				if (used[c2]) continue;
				double d = D1(c1, c2);
				if (d < best)
				{
					best = d;
					best_c2 = c2;
				}
			}
			mapping[c1] = best_c2;
			if (best_c2 >= 0) used[best_c2] = 1;
		}

		// set ub_sol rows via labels
		arma::uvec labels = arma::index_max(sol, 1);
		for (arma::uword r = 0; r < nh; ++r)
		{
			int c = static_cast<int>(labels(r));
			int a = mapping[c];
			if (a >= 0) ub_sol(idx(r), a) = 1.0;
		}

		// write objective row in mapped order and cent_diff
		for (int c1 = 0; c1 < k; ++c1) {
			int a = mapping[c1];
			obj(h, a) = obj_h(c1);
		}
	}

	// save new sol as points and sizes
	for (int h = 0; h < p; ++h)
	{
		arma::uvec idx_h = arma::find(antic_sol.col(h) == 1.0);
		for (int c = 0; c < k; ++c)
		{
			arma::uvec sel = arma::find((antic_sol.col(h) == 1.0) % (ub_sol.col(c) == 1.0));
			points[c][h].clear();
			points[c][h].reserve(sel.n_elem);
			for (arma::uword t = 0; t < sel.n_elem; ++t)
				points[c][h].push_back(static_cast<int>(sel(t)));
			sizes[c][h] = static_cast<int>(sel.n_elem);
		}
	}

	return obj;
}

// Update points and sizes after a swap to evaluate
void evaluate_swap(arma::mat& data,
				   arma::mat& antic_sol,
				   arma::mat& centroids_heu,
				   arma::mat& new_sol,
				   const std::vector<int>& hList,
				   arma::mat& new_W_hc)
{
	for (int h : hList)
	{
		// gather indices of points in partition h
		arma::uvec idx = arma::find(antic_sol.col(h) == 1.0);
		arma::mat data_antic = data.rows(idx);

		// run k-means with provided centroids init
		Kmeans km(data_antic, k, kmeans_verbose);
		km.start(kmeans_max_it, 1, centroids_heu);
		new_W_hc.row(h) = km.objectiveFunctionCls().t();
		arma::uvec labels = arma::index_max(km.getAssignments(), 1);

		// create new solution
		new_sol.rows(idx).zeros();
		for (arma::uword r = 0; r < idx.n_elem; ++r)
		{
			new_sol(idx(r), labels(r)) = 1.0;
		}
	}
}

// update points and sizes after a swap
void update_sol(arma::mat& antic_sol,
				arma::mat& ub_sol,
				std::vector<std::vector<std::vector<int>>>& points,
				std::vector<std::vector<int>>& sizes,
				const std::vector<int>& hList)
{
	for (int c = 0; c < k; c++)
	{
		for (int h : hList)
		{
			points[c][h].clear();
			points[c][h].reserve(n);
			int nc = 0;
			for (int i = 0; i < n; i++)
				if (antic_sol(i, h) == 1 and ub_sol(i, c) == 1)
				{
					points[c][h].push_back(i);
					nc++;
				}
			sizes[c][h] = nc;
		}
	}
}


// Use SOS-SDP to compute lower and upper bounds for the anticlustering problem
std::pair<double, double> compute_lb(arma::mat& data, const std::vector<std::vector<int>>& sol, bool verbose)
{
	auto* shared = new SharedDataPartition();
	shared->lb_part.assign(p, 0.0);
	shared->ub_part.assign(p, 0.0);
	shared->sol_part.resize(p);
	shared->threadStates.reserve(n_threads_part);
	for (int i = 0; i < n_threads_part; ++i) shared->threadStates.push_back(false);

	for (int h = 0; h < p; ++h)
	{
		auto* job = new PartitionJob();
		job->part_id = h;

		// build data_part via index vector (no row-by-row copy)
		arma::uvec idx_h(sol[h].size());
		for (arma::uword t = 0; t < idx_h.n_elem; ++t) idx_h(t) = static_cast<arma::uword>(sol[h][t]);
		arma::mat data_part = data.rows(idx_h);
		save_to_file(data_part, "pf_" + std::to_string(h));

		// call python to compute k-means ub with improved heuristic
		std::string str_path = result_path + "_pf_" + std::to_string(h) + ".txt";
		std::string out_amnt = result_path + "_out_" + std::to_string(h) + ".txt";
		std::string args = str_path + " " + std::to_string(k) + " " + out_amnt;
		std::string command = "python3 ./run_kmeans.py " + args + " 1";
		std::cout << command << "\n";
		int system_val = system(command.c_str());
		if (system_val == -1)
		{
			// The system method failed
			std::cout << "Failed to call Python script" << "\n";
			exit(EXIT_FAILURE);
		}

		job->part_data = data_part;
		job->kmeans_ub = read_part_val(out_amnt.c_str());
		shared->queue.push_back(job);
		shared->print = verbose;
	}

	// solve with SOS-SDP
	ThreadPoolPartition p_pool(shared, n_threads_part);
	while (true)
	{
		{
			std::unique_lock<std::mutex> l(shared->queueMutex);
			while (is_thread_pool_working(shared->threadStates))
			{
				shared->mainConditionVariable.wait(l);
			}
			if (shared->queue.empty()) break;
		}
	}

	// collect all the results
	double lb_mss = 0.0;
	double ub_mss = 0.0;
	for (auto& v : shared->lb_part) lb_mss += v;
	for (auto& v : shared->ub_part) ub_mss += v;

	// free memory
	p_pool.quitPool();
	delete (shared);

	return {lb_mss, ub_mss};
}

// run the anticlustering heuristic
void avoc(arma::mat& data, HResult& results)
{
	std::cout << "\nRunning heuristics anticluster with k-means esteeme..\n\n";

	// -------------------- Create initial partition and heuristic centroids --------------------
	std::cout << "\nCreating starting anticlustering partition\n\n";
	std::vector<std::vector<std::vector<int>>> sol_cls(k, std::vector<std::vector<int>>(p));
	std::vector<std::vector<int>> init_sol(p);

	// compute heuristic centroids from initial solution
	std::vector<arma::uvec> idx_by_c(k);
	for (int c = 0; c < k; ++c) idx_by_c[c] = arma::find(results.init_sol.col(c) == 1.0);
	arma::mat centroids_heu(k, d, arma::fill::zeros);
	for (int c = 0; c < k; ++c) centroids_heu.row(c) = arma::mean(data.rows(idx_by_c[c]), 0);

	// compute distances matrix
	SymmDist all_dist(n);
	for (int i = 0; i < n; ++i)
	{
		const arma::rowvec xi = data.row(i);
		for (int j = i + 1; j < n; ++j)
		{
			const arma::rowvec diff = xi - data.row(j);
			//all_dist.at(i, j) = static_cast<float>(arma::dot(diff, diff)); // squared Euclidean
			all_dist.at(i, j) = std::pow(arma::norm(data.row(i).t() - data.row(j).t(), 2),2); // squared Euclidean
			double sq_dist = arma::dot(diff, diff);
			all_dist.at(i, j) = sq_dist * sq_dist;
		}
	}

	GRBEnv* env = new GRBEnv();
	auto* mdl = new mount_gurobi_model(env, k * (k - 1) * p * p / 2);
	mdl->add_point_constraints();
	mdl->add_cls_constraints();
	mdl->add_edge_constraints();

	std::random_device rd;
	std::mt19937 gen(rd());
	double best_val = std::numeric_limits<double>::infinity();
	arma::mat antic_sol_best;
	auto start_time_m = std::chrono::steady_clock::now();

	for (int s = 0; s < n_starts; ++s)
	{
		// Random round-robin split inside each original cluster c → p buckets
		for (int c = 0; c < k; ++c)
		{
			const arma::uvec& idx_c = idx_by_c[c];

			std::vector<int> tmp(idx_c.n_elem);
			for (arma::uword r = 0; r < idx_c.n_elem; ++r) tmp[r] = static_cast<int>(idx_c(r));
			std::shuffle(tmp.begin(), tmp.end(), gen);

			int q = tmp.size() / p;  // base size
			int r = tmp.size() % p;  // remainder
			size_t start = 0;
			for (int h = 0; h < p; ++h) {
				int sz = q + (h < r ? 1 : 0);
				sol_cls[c][h].assign(tmp.begin() + start, tmp.begin() + start + sz);
				start += sz;
			}
		}

		try
		{
			mdl->update_Y_objective(all_dist, sol_cls);
			mdl->reset();
			mdl->optimize();

			init_sol = mdl->get_x_solution(sol_cls);
		}
		catch (GRBException& e)
		{
			std::cout << "Error code = " << e.getErrorCode() << "\n";
			std::cout << e.getMessage() << "\n";
		}

		auto [obj, cen] = evaluate_partition_idx(data, init_sol, centroids_heu);
		if (cen < best_val)
		{
			best_val = cen;
			antic_sol_best.set_size(n, p);
			antic_sol_best.zeros();
			for (int h = 0; h < p; ++h)
				for (int i : init_sol[h]) antic_sol_best(i, h) = 1.0;
		}
	}

	delete mdl;
	delete env;

	// save mount time
	auto end_time_m = std::chrono::steady_clock::now();
	results.m_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_m - start_time_m).count();

	arma::mat& antic_sol = antic_sol_best;

	// -------------------- Build first anticluster solution --------------------
	// create first ub sol via Python-based k-means per partition (kept)
	std::vector<std::vector<std::vector<int>>> points(k, std::vector<std::vector<int>>(p));
	std::vector<std::vector<int>> sizes(k, std::vector<int>(p, 0));
	arma::mat ub_sol(n, k, arma::fill::zeros);
	arma::mat W_hc = create_first_sol(data, antic_sol, centroids_heu, ub_sol, points, sizes);

	std::vector<std::vector<int>> first_sol(p);
	for (int h = 0; h < p; h++)
	{
		first_sol[h].reserve(n);
		for (int c = 0; c < k; c++)
			for (int i = 0; i < sizes[c][h]; i++)
			{
				first_sol[h].push_back(points[c][h][i]);
			}
	}
	results.first_mss = arma::accu(W_hc);

	// -------------------- Swap heuristic --------------------
	auto start_time_h = std::chrono::steady_clock::now();
	double best_W = arma::accu(W_hc);
	double best_GAP = (results.heu_mss - best_W) * 100.0 / results.heu_mss;

	log_file << "\n\nAVOC Algorithm\n"
		<< "Initial LB: " << std::setprecision(2) << best_W << "\n"
		<< "Initial GAP: " << std::setprecision(4) << best_GAP << "\n";

	log_file << "\n\nIt | k | LB+ | GAP x100";
	log_file << "\n 0 | - | " << std::setprecision(2) <<
		best_W << " | " << std::setprecision(4) << best_GAP;
	std::printf("\n\nAnticlustering Heuristic\nIt | k | LB+ | GAP x100");
	std::printf("\n%d | %d | %.2f | %.4f", 0, 0, best_W, best_GAP);

	results.it = 0;
	results.n_swaps = 0;

	while (true)
	{
		results.it++;

		if (best_GAP < min_gap)
		{
			log_file << "\n\nMin GAP " << std::setprecision(3) << min_gap << " reached\n\n";
			std::cout << "\n\nMin GAP " << std::setprecision(3) << min_gap << " reached\n\n";
			results.stop = "tol";
			break;
		}

		// check time limit
		if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_h).count() > (max_time * p))
		{
			log_file << "\n\nTime limit reached\n\n";
			std::cout << "\n\nTime limit reached\n\n";
			results.stop = "time";
			break;
		}

		bool found_better = false;

		for (int c = 0; c < k; c++)
		{
			arma::uvec worst_p = arma::sort_index(W_hc.col(c), "ascend");
			for (int idx1 = 0; idx1 < p; idx1++)
			{
				bool found_better_ch = false;

				// Compute centroid to select nearest point
				int h1 = worst_p(idx1);
				int idx_p1 = 0;
				int p1 = points[c][h1][idx_p1];
				for (int idx2 = p - 1; idx2 >= 0 && !found_better_ch; idx2--)
				{
					if (idx1 != idx2)
					{
						int h2 = worst_p(idx2);
						for (int idx_p2 = points[c][h2].size() - 1; idx_p2 >= 0 && !found_better_ch; idx_p2--)
						{
							int p2 = points[c][h2][idx_p2];

							// evaluate swap
							arma::mat new_sol = ub_sol;
							arma::mat new_antic_sol = antic_sol;
							new_antic_sol(p1, h1) = 0;
							new_antic_sol(p1, h2) = 1;
							new_antic_sol(p2, h1) = 1;
							new_antic_sol(p2, h2) = 0;
							std::vector<int> hList;
							hList.push_back(h1);
							hList.push_back(h2);
							arma::mat new_W_hc = W_hc;
							evaluate_swap(data, new_antic_sol, centroids_heu, new_sol, hList, new_W_hc);
							double W = arma::accu(new_W_hc);
							if (W > best_W)
							{
								found_better = true;
								found_better_ch = true;
								best_W = W;
								W_hc = new_W_hc;
								ub_sol = new_sol;
								antic_sol = new_antic_sol;
								update_sol(antic_sol, ub_sol, points, sizes, hList);
								best_GAP = (results.heu_mss - best_W) * 100 / results.heu_mss;

								results.n_swaps++;
								log_file << "\n" << results.it << " | " << c + 1 << " | " << std::setprecision(2) <<
									best_W << " | " << std::setprecision(4) << best_GAP;
								std::printf("\n%d | %d | %.2f | %.3f", results.it, c + 1, best_W, best_GAP);
							}
						}
					}
				}
			}
		}

		if (!found_better)
		{
			log_file << "\n\nNo improvement -> stop (stall)\n\n";
			std::cout << "\n\nNo improvement -> stop (stall)\n\n";
			results.stop = "stall";
			break;
		}
	}

	// save heuristic time
	auto end_time_h = std::chrono::steady_clock::now();
	results.h_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_h - start_time_h).count();

	results.anti_obj = best_W;
	std::vector<std::vector<int>> sol(p);
	for (int h = 0; h < p; h++)
	{
		sol[h].reserve(n);
		for (int c = 0; c < k; c++)
			for (int i = 0; i < sizes[c][h]; i++)
			{
				sol[h].push_back(points[c][h][i]);
			}
	}
	log_file << "\n---------------------------------------------------------------------\n";

	// -------------------- Compute true lower bound --------------------
	auto start_time_lb = std::chrono::steady_clock::now();
	log_file << "\n\n" << "|" <<
			  std::setw(5) << "N" << "|" <<
			  std::setw(9) << "NODE_PAR" << "|" <<
			  std::setw(8) << "NODE" << "|" <<
			  std::setw(12) << "LB_PAR" << "|" <<
			  std::setw(12) << "LB" << "|" <<
			  std::setw(6) << "FLAG" << "|" <<
			  std::setw(10) << "TIME (s)" << "|" <<
			  std::setw(8) << "CP_ITER" << "|" <<
			  std::setw(8) << "CP_FLAG" << "|" <<
			  std::setw(10) << "CP_INEQ" << "|" <<
			  std::setw(9) << "PAIR" << " " <<
			  std::setw(9) << "TRIANGLE" << " " <<
			  std::setw(9) << "CLIQUE" << "|" <<
			  std::setw(12) << "UB" << "|" <<
			  std::setw(12) << "GUB" << "|" <<
			  std::setw(6) << "I" << " " <<
			  std::setw(6) << "J" << "|" <<
			  std::setw(13) << "NODE_GAP (%)" << "|" <<
			  std::setw(13) << "GAP (%)" << "|" <<
			  std::setw(6) << "OPEN" << "|"
			  << "\n";

	auto [fst, snd] = compute_lb(data, sol, true);
	results.lb_mss = fst;
	results.ub_mss = snd;

	// save lower bound time
	auto end_time_lb = std::chrono::steady_clock::now();
	results.lb_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_lb - start_time_lb).count();

	// -------------------- Save results --------------------
	std::cout << "\n*********************************************************************\n";

	log_file << "\n*********************************************************************";
	log_file << "\nInit sol: " << results.heu_mss;
	log_file << "\nFirst LB: " << results.first_mss;
	log_file << "\nANTI OBJ: " << results.anti_obj;
	log_file << "\nLB: " << results.lb_mss;
	log_file << "\nLB+: " << results.ub_mss;
	log_file << "\nSWAPS: " << results.n_swaps;
	log_file << "\nSTOP: " << results.stop;
	log_file << "\nGAP_LB' " << (results.heu_mss - results.first_mss) * 100 / results.heu_mss;
	log_file << "\nGAP^+ " << (results.heu_mss - results.anti_obj) * 100 / results.heu_mss;
	log_file << "\nGAP_LB " << (results.heu_mss - results.lb_mss) * 100 / results.heu_mss;
	log_file << "\nGAP_UB+ " << (results.heu_mss - results.ub_mss) * 100 / results.heu_mss;
	log_file << "\n*********************************************************************\n\n";
}
