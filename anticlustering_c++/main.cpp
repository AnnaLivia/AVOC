#include <iostream>
#include <filesystem>
#include <map>
#include <algorithm>
#include <armadillo>
#include "ac_heuristic.h"

// data file
const char *data_path;
const char *init_path;

// log and result path
std::string inst_name;
std::string result_folder;
std::string result_path;
std::ofstream log_file;
std::mutex file_mutex;
std::ofstream lb_file;

// instance data
int n;
int d;
int p;
int k;

// partition and anticlustering
int n_starts;
double max_time;
double min_gap;
int n_threads_part;

// k-means
int kmeans_max_it;
int kmeans_start;
bool kmeans_verbose;

// branch and bound
double branch_and_bound_tol;
int branch_and_bound_parallel;
int branch_and_bound_max_nodes;
int branch_and_bound_visiting_strategy;

// sdp solver
int sdp_solver_session_threads_root;
int sdp_solver_session_threads;
const char *sdp_solver_folder;
double sdp_solver_tol;
int sdp_solver_stopoption;
int sdp_solver_maxiter;
int sdp_solver_maxtime;
int sdp_solver_verbose;
int sdp_solver_max_cp_iter_root;
int sdp_solver_max_cp_iter;
double sdp_solver_cp_tol;
int sdp_solver_max_ineq;
double sdp_solver_inherit_perc;
double sdp_solver_eps_ineq;
double sdp_solver_eps_active;
int sdp_solver_max_pair_ineq;
double sdp_solver_pair_perc;
int sdp_solver_max_triangle_ineq;
double sdp_solver_triangle_perc;

// read parameters in config file
std::map<std::string, std::string> read_params(const std::string& path) {
	std::map<std::string, std::string> cfg;
	std::ifstream in(path);
	if (!in) { std::cerr << "Couldn't open config file: " << path << "\n"; return cfg; }

	std::string line;
	while (std::getline(in, line)) {
		// trim spaces and tabs
		line.erase(std::remove_if(line.begin(), line.end(),
								  [](unsigned char c){ return c==' ' || c=='\t' || c=='\r'; }),
				   line.end());
		if (line.empty() || line[0]=='#') continue;
		auto pos = line.find('=');
		if (pos==std::string::npos || pos==0 || pos+1>=line.size()) continue;
		cfg.emplace(line.substr(0,pos), line.substr(pos+1));
	}
	return cfg;
}

// save sorted clustering solution and data to file
void sort_all(arma::mat& data, arma::mat& sol) {
	const arma::uword N = data.n_rows, D = data.n_cols, K = sol.n_cols;

	// Compute centroids
	arma::vec counts = arma::sum(sol, 0).t();     // K x 1
	arma::mat centroids = sol.t() * data;         // K x D
	centroids.each_col() /= counts;

	// labels for each point (N x 1) in {0..K-1}
	arma::uvec labels = arma::index_max(sol, 1);

	// squared distance of each point to its assigned centroid
	arma::mat picked = centroids.rows(labels);    // N x D (gathers centroid row per point)
	arma::mat diff = data - picked;
	arma::vec dist2 = arma::sum(arma::square(diff), 1);  // N x 1

	// sort by ascending distance
	arma::uvec order = arma::sort_index(dist2, "ascend");
	data = data.rows(order);
	sol  = sol.rows(order);

	//save_to_file(sol, "KM");
	//save_to_file(data, "DATA");
}

// write log preamble for instance
void write_log_preamble(double init_mss) {

	log_file << "DATA_FILE, n, d, k, t:\n";
    log_file << data_path << " " << n << " " << d << " " << k << " " << p << "\n\n";

    log_file << "ALGORITHM_INIT_MULTI_START: " << n_starts << "\n";
    log_file << "ALGORITHM_MIN_GAP_CRITERION: " <<  std::setprecision(3) << min_gap << "\n";
    log_file << "ALGORITHM_MAX_TIME_PER_ANTI_CRITERION: " <<  std::setprecision(3) << max_time << "\n";
	log_file << "PARTITION_THREADS: " << n_threads_part << "\n\n";

	log_file << "KMEANS_NUM_START: " << kmeans_start << "\n";
	log_file << "KMEANS_MAX_ITERATION: " << kmeans_max_it << "\n";
	log_file << "KMEANS_VERBOSE: " << kmeans_verbose << "\n\n";

    log_file << "BRANCH_AND_BOUND_TOL: " << branch_and_bound_tol << "\n";
    log_file << "BRANCH_AND_BOUND_PARALLEL: " << branch_and_bound_parallel << "\n";
    log_file << "BRANCH_AND_BOUND_MAX_NODES: " << branch_and_bound_max_nodes << "\n";
    log_file << "BRANCH_AND_BOUND_VISITING_STRATEGY: " << branch_and_bound_visiting_strategy << "\n\n";

    log_file << "SDP_SOLVER_SESSION_THREADS_ROOT: " << sdp_solver_session_threads_root << "\n";
    log_file << "SDP_SOLVER_SESSION_THREADS: " << sdp_solver_session_threads << "\n";
    log_file << "SDP_SOLVER_FOLDER: " << sdp_solver_folder << "\n";
    log_file << "SDP_SOLVER_TOL: " << sdp_solver_tol << "\n";
    log_file << "SDP_SOLVER_VERBOSE: " << sdp_solver_verbose << "\n";
    log_file << "SDP_SOLVER_MAX_CP_ITER_ROOT: " << sdp_solver_max_cp_iter_root << "\n";
    log_file << "SDP_SOLVER_MAX_CP_ITER: " << sdp_solver_max_cp_iter << "\n";
    log_file << "SDP_SOLVER_CP_TOL: " << sdp_solver_cp_tol << "\n";
    log_file << "SDP_SOLVER_MAX_INEQ: " << sdp_solver_max_ineq << "\n";
    log_file << "SDP_SOLVER_INHERIT_PERC: " << sdp_solver_inherit_perc << "\n";
    log_file << "SDP_SOLVER_EPS_INEQ: " << sdp_solver_eps_ineq << "\n";
    log_file << "SDP_SOLVER_EPS_ACTIVE: " << sdp_solver_eps_active << "\n";
    log_file << "SDP_SOLVER_MAX_PAIR_INEQ: " << sdp_solver_max_pair_ineq << "\n";
    log_file << "SDP_SOLVER_PAIR_PERC: " << sdp_solver_pair_perc << "\n";
    log_file << "SDP_SOLVER_MAX_TRIANGLE_INEQ: " << sdp_solver_max_triangle_ineq << "\n";
    log_file << "SDP_SOLVER_TRIANGLE_PERC: " << sdp_solver_triangle_perc << "\n\n";
    log_file << std::fixed <<  std::setprecision(2);

	log_file << "\n---------------------------------------------------------------------\n";
    log_file << "UB: " << init_mss << "\n";

}

// read data Ws
arma::mat read_data(const char *filename) {

    std::ifstream file(filename);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }

    // read the header n, d
    file >> n >> d;
    arma::mat data(n, d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            file >> data(i, j);
        }
    }

    return data;
}

// read clustering sol
arma::mat read_sol(const char* filename) {
	std::ifstream f(filename);
	if (!f) { std::cerr << strerror(errno) << "\n"; std::exit(EXIT_FAILURE); }

	// peek first line to infer columns
	std::streampos start = f.tellg();
	std::string first;
	std::getline(f, first);
	std::istringstream iss(first);
	int cols = 0; double tmp;
	while (iss >> tmp) ++cols;

	// rewind
	f.clear();
	f.seekg(start);

	arma::mat sol(n, k, arma::fill::zeros);

	if (cols == 1) {
		// label per row
		for (int i = 0; i < n; ++i) {
			int c; f >> c;
			if (c < 0 || c >= k) { std::cerr << "read_sol(): label out of range\n"; std::exit(EXIT_FAILURE); }
			sol(i, c) = 1.0;
		}
	} else {
		if (cols != k) { std::cerr << "read_sol(): expected " << k << " columns, got " << cols << "\n"; std::exit(EXIT_FAILURE); }
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < k; ++j)
				f >> sol(i, j);
	}
	return sol;
}

// compute mss for a given solution
double compute_mss(const arma::mat& data, const arma::mat& sol) {
	const arma::uword N = data.n_rows;

	// counts per cluster
	arma::vec counts = arma::sum(sol, 0).t();
	if (arma::any(counts == 0)) {
		std::cerr << strerror(errno) << "\n";
		exit(EXIT_FAILURE);
	}

	// centroids (K x D)
	arma::mat centroids = sol.t() * data;
	centroids.each_col() /= counts;

	// label of each point (N x 1)
	arma::uvec labels = arma::index_max(sol, 1);

	// pick each point's centroid (N x D), compute row-wise squared distances
	arma::mat picked = centroids.rows(labels);
	arma::mat diff   = data - picked;
	arma::vec row_sq = arma::sum(arma::square(diff), 1);

	return arma::accu(row_sq);
}

void run(int argc, char **argv) {

    std::string config_file = "config.txt";
    std::map <std::string, std::string> config_map = read_params(config_file);

    result_folder = config_map["RESULT_FOLDER"];

    // number of thread for computing the partition bound
    n_starts = std::stoi(config_map["ALGORITHM_INIT_MULTI_START"]);
    min_gap = std::stod(config_map["ALGORITHM_MIN_GAP_CRITERION"]);
    max_time = std::stoi(config_map["ALGORITHM_MAX_TIME_PER_ANTI_CRITERION"]);
    n_threads_part = std::stoi(config_map["PARTITION_THREADS"]);

	// kmeans
	kmeans_start = std::stoi(config_map["KMEANS_NUM_START"]);
	kmeans_max_it = std::stoi(config_map["KMEANS_MAX_ITERATION"]);
	kmeans_verbose = std::stoi(config_map["KMEANS_VERBOSE"]);

    // branch and bound
    branch_and_bound_tol = std::stod(config_map["BRANCH_AND_BOUND_TOL"]);
    branch_and_bound_parallel = std::stoi(config_map["BRANCH_AND_BOUND_PARALLEL"]);
    branch_and_bound_max_nodes = std::stoi(config_map["BRANCH_AND_BOUND_MAX_NODES"]);
    branch_and_bound_visiting_strategy = std::stoi(config_map["BRANCH_AND_BOUND_VISITING_STRATEGY"]);

    // sdp solver
    sdp_solver_session_threads_root = std::stoi(config_map["SDP_SOLVER_SESSION_THREADS_ROOT"]);
    sdp_solver_session_threads = std::stoi(config_map["SDP_SOLVER_SESSION_THREADS"]);
    sdp_solver_folder = config_map["SDP_SOLVER_FOLDER"].c_str();
    sdp_solver_tol = std::stod(config_map["SDP_SOLVER_TOL"]);
    sdp_solver_verbose = std::stoi(config_map["SDP_SOLVER_VERBOSE"]);
    sdp_solver_max_cp_iter_root = std::stoi(config_map["SDP_SOLVER_MAX_CP_ITER_ROOT"]);
    sdp_solver_max_cp_iter = std::stoi(config_map["SDP_SOLVER_MAX_CP_ITER"]);
    sdp_solver_cp_tol = std::stod(config_map["SDP_SOLVER_CP_TOL"]);
    sdp_solver_max_ineq = std::stoi(config_map["SDP_SOLVER_MAX_INEQ"]);
    sdp_solver_inherit_perc = std::stod(config_map["SDP_SOLVER_INHERIT_PERC"]);
    sdp_solver_eps_ineq = std::stod(config_map["SDP_SOLVER_EPS_INEQ"]);
    sdp_solver_eps_active = std::stod(config_map["SDP_SOLVER_EPS_ACTIVE"]);
    sdp_solver_max_pair_ineq = std::stoi(config_map["SDP_SOLVER_MAX_PAIR_INEQ"]);
    sdp_solver_pair_perc = std::stod(config_map["SDP_SOLVER_PAIR_PERC"]);
    sdp_solver_max_triangle_ineq = std::stoi(config_map["SDP_SOLVER_MAX_TRIANGLE_INEQ"]);
    sdp_solver_triangle_perc = std::stod(config_map["SDP_SOLVER_TRIANGLE_PERC"]);
    sdp_solver_stopoption = 0;
    sdp_solver_maxiter = 50000;
    sdp_solver_maxtime = 3*3600;
    
    if (argc != 5) {
        std::cerr << "Input: <DATA_FILE> <SOL_FILE> <K> <T>\n";
        exit(EXIT_FAILURE);
    }

    // read input parameters
    data_path = argv[1];
    init_path = argv[2];

    k = std::stoi(argv[3]);
    p = std::stoi(argv[4]);

	if (k <= 0 || p <= 0) {
		std::cerr << "K and T must be positive.\n";
		std::exit(EXIT_FAILURE);
	}

	unsigned hw = std::max(1u, std::thread::hardware_concurrency());
	n_threads_part = std::clamp(n_threads_part, 1, (int)hw);

	// percentage gap
	auto pct = [](double ub, double lb){
		if (ub <= 0) return 0.0;
		return (ub - lb) * 100.0 / ub;
	};

	// read data and init sol
    std::string str_path = data_path;
    inst_name = str_path.substr(str_path.find_last_of("/\\")+1);
    inst_name = inst_name.substr(0, inst_name.find("."));
    std::ofstream test_SUMMARY(result_folder.substr(0, result_folder.find_last_of("/\\")) + "/test_SUMMARY.txt", std::ios::app);

    result_path = result_folder + "/" + std::to_string(p) + "part/" + inst_name;
    if (!std::filesystem::exists(result_path))
        std::filesystem::create_directories(result_path);
    result_path += "/" + inst_name + "_" + std::to_string(k);

    arma::mat Ws = read_data(data_path);

    // read init sol
	arma::mat init_sol = read_sol(init_path);
    double init_mss = compute_mss(Ws, init_sol);
	sort_all(Ws, init_sol);

	// printing instance info
    std::cout << "\n---------------------------------------------------------------------\n";
    std::cout << "Instance "       		   << inst_name << "\n";
    std::cout << "Num Points " 	   		   << n << "\n";
    std::cout << "Num Features "   		   << d << "\n";
    std::cout << "Num Partitions " 		   << p << "\n";
    std::cout << "Num Clusters "   		   << k << "\n\n";
    std::cout << std::fixed 	   		   << std::setprecision(2);
    std::cout << "Initial Heuristic MSS: " << init_mss << "\n";
    std::cout << "---------------------------------------------------------------------\n\n";
    log_file.open(result_path + "_LOGinit.txt");
    write_log_preamble(init_mss);

    test_SUMMARY << std::fixed << std::setprecision(2)
    << inst_name << "\t"
    << n << "\t"
    << d << "\t"
    << k << "\t"
    << p << "\t"
    << init_mss << "\t";
    test_SUMMARY << std::fixed << std::setprecision(3);

	// Run the anticlustering heuristic
    HResult results;
    results.init_sol = init_sol;
    results.heu_mss = init_mss;
    results.first_mss = 0;
    results.lb_time = 0;
    results.h_time = 0;
    results.m_time = 0;

    avoc(Ws, results);

    std::cout << "\nINIT: "    << results.heu_mss << "\n";
    std::cout << "FIRST LB+: " << results.first_mss << "\n";
    std::cout << "ANTI OBJ: "  << results.anti_obj << "\n";
    std::cout << "LB: "        << results.lb_mss << "\n";
    std::cout << "LB+: " 	   << results.ub_mss << "\n";
    std::cout << "SWAPS: " 	   << results.n_swaps << "\n";
    std::cout << "STOP: " 	   << results.stop << "\n";
    std::cout << "---------------------------------------------------------------------\n";
    std::cout << "GAP_LB' " << pct(results.heu_mss, results.first_mss) << "%\n";
	std::cout << "---------------------------------------------------------------------\n";
	std::cout << "GAP^+ " 		 << pct(results.heu_mss, results.anti_obj) << "%\n";
	std::cout << "---------------------------------------------------------------------\n";
    std::cout << "GAP_LB " 		 << pct(results.heu_mss, results.lb_mss) << "%\n";
    std::cout << "---------------------------------------------------------------------\n";
    std::cout << "GAP_U " 		 << pct(results.heu_mss, results.ub_mss) << "%\n";
    std::cout << "---------------------------------------------------------------------\n";

	test_SUMMARY
	<< results.first_mss << "\t"
	<< results.anti_obj  << "\t"
	<< results.ub_mss    << "\t"
	<< results.lb_mss    << "\t"
    << pct(results.heu_mss, results.first_mss) << "%\t"
	<< pct(results.heu_mss, results.anti_obj) << "%\t"
    << pct(results.heu_mss, results.lb_mss) << "%\t"
	<< pct(results.heu_mss, results.ub_mss) << "%\t"
    << results.m_time    << "\t"
    << results.h_time    << "\t"
    << results.n_swaps   << "\t"
    << results.stop      << "\t"
    << results.lb_time   << "\t"
    << (results.h_time + results.m_time + results.lb_time)/60    << "\n";

    log_file.close();
	test_SUMMARY.close();

}

int main(int argc, char **argv) {

    run(argc, argv);

    return EXIT_SUCCESS;
}
