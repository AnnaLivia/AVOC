#include <iostream>
#include <filesystem>
#include <map>
#include <algorithm>
#include <armadillo>
#include "Kmeans.h"
#include "kmeans_util.h"
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
double max_time;
double min_gap;
int n_threads_anti;
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
std::map<std::string, std::string> read_params(std::string &config_file) {

    std::map<std::string, std::string> config_map = {};

    std::ifstream cFile (config_file);
    if (cFile.is_open()) {
        std::string line;
        while (getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if(line[0] == '#' || line.empty())
                continue;
            auto delimiterPos = line.find('=');
            auto key = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);
            config_map.insert(std::pair<std::string, std::string>(key, value));
        }

    }
    else {
        std::cerr << "Couldn't open config file for reading.\n";
    }

    return config_map;
}

void sort_all(arma::mat &data, arma::mat &sol) {

    //reorder data e init_sol
    arma::mat centroids(k, d);
    arma::vec count(k);
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < k; ++c)
            if (sol(i, c) == 1) {
        		centroids.row(c) += data.row(i);
        		count(c) += 1;
        	}
    }
    for (int c = 0; c < k; ++c)
    	centroids.row(c) /= count(c);

    arma::vec distances(n);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < k; ++c)
            if (sol(i, c) == 1) {
                distances(i) = std::pow(norm(data.row(i) - centroids.row(c), 2), 2);
                break;
            }

	// Sort distances and get the sorted indices
    arma::uvec sorted_indices = arma::sort_index(distances, "ascend");
    arma::mat sorted_data = data.rows(sorted_indices);
    arma::mat sorted_sol = sol.rows(sorted_indices);

    data = sorted_data;
    sol = sorted_sol;
    save_to_file(sol, "KM");
    save_to_file(data, "DATA");

}

// write log preamble for instance
void write_log_preamble(double init_mss) {

	log_file << "DATA_FILE, n, d, k, t:\n";
    log_file << data_path << " " << n << " " << d << " " << k << " " << p << "\n\n";

    log_file << "ALGORITHM_MIN_GAP_CRITERION: " <<  std::setprecision(3) << min_gap << "\n";
    log_file << "ALGORITHM_MAX_TIME_PER_ANTI_CRITERION: " <<  std::setprecision(3) << max_time << "\n";
    log_file << "ANTICLUSTERING_THREADS: " << n_threads_anti << "\n";
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
arma::mat read_sol(const char *filename) {

	std::ifstream file(filename);
	if (!file) {
		std::cerr << strerror(errno) << "\n";
		exit(EXIT_FAILURE);
	}

	std::ifstream filecheck(filename);
	std::string line;
	getline(filecheck, line);
	std::stringstream ss(line);
	int cols = 0;
	double item;
	while(ss >> item) cols++;
	filecheck.close();

	arma::mat sol(n, k);
	for (int i = 0; i < n; i++) {
		if (cols == 1) {
			sol.row(i) = arma::zeros(k).t();
			int c = 0;
			file >> c;
			sol(i, c) = 1;
		}
		else {
			for (int j = 0; j < k; j++)
				file >> sol(i, j);
		}
	}

	return sol;
}


// compute mss for a given solution
double compute_mss(arma::mat &data, arma::mat &sol) {

    int data_n = data.n_rows;
    int data_d = data.n_cols;
    int data_k = sol.n_cols;

    if (data_n != sol.n_rows) {
        std::printf("compute_mss() - ERROR: inconsistent data and sol!\n");
        exit(EXIT_FAILURE);
    }

    arma::mat assignment_mat = arma::zeros(data_n, data_k);
    arma::vec count = arma::zeros(data_k);
    arma::mat centroids = arma::zeros(data_k, data_d);
    for (int i = 0; i < data_n; i++) {
        for (int j = 0; j < data_k; j++) {
            if (sol(i,j) == 1) {
                assignment_mat(i, j) = 1;
                ++count(j);
                centroids.row(j) += data.row(i);
            }
        }
    }

    // compute clusters' centroids
    for (int c = 0; c < data_k; c++) {
        // empty cluster
        if (count(c) == 0) {
            std::printf("compute_mss(): cluster %d is empty!\n", c);
            return false;
        }
        centroids.row(c) = centroids.row(c) / count(c);
    }

    arma::mat m = data - assignment_mat * centroids;

    return arma::dot(m.as_col(), m.as_col());
}

void run(int argc, char **argv) {

    std::string config_file = "config.txt";
    std::map <std::string, std::string> config_map = read_params(config_file);

    result_folder = config_map["RESULT_FOLDER"];

    // number of thread for computing the partition bound
    min_gap = std::stod(config_map["ALGORITHM_MIN_GAP_CRITERION"]);
    max_time = std::stoi(config_map["ALGORITHM_MAX_TIME_PER_ANTI_CRITERION"]);
    n_threads_anti = std::stoi(config_map["ANTICLUSTERING_THREADS"]);
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
        std::cerr << "Input: <DATA_FILE> <SOL_FILE> <K> <T>" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    data_path = argv[1];
    init_path = argv[2];

    k = std::stoi(argv[3]);
    p = std::stoi(argv[4]);

    if (k < n_threads_anti)
        n_threads_anti = k;
    if (p < n_threads_part)
        n_threads_part = p;

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

    std::cout << std::endl << "---------------------------------------------------------------------" << std::endl;
    std::cout << "Instance " << inst_name << std::endl;
    std::cout << "Num Points " << n << std::endl;
    std::cout << "Num Features " << d << std::endl;
    std::cout << "Num Partitions " << p << std::endl;
    std::cout << "Num Clusters " << k << std::endl << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Initial Heuristic MSS: " << init_mss << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl << std::endl;

    log_file.open(result_path + "_LOG.txt");
    write_log_preamble(init_mss);

    double best_lb = 0;
    double best_lb_plus = 0;
    double best_ub = init_mss;

	// printing results
    test_SUMMARY << std::fixed << std::setprecision(2)
    << inst_name << "\t"
    << n << "\t"
    << d << "\t"
    << k << "\t"
    << p << "\t"
    << init_mss << "\t";

    test_SUMMARY << std::fixed << std::setprecision(3);
	
    HResult results;
    results.init_sol = init_sol;
    results.heu_mss = init_mss;
    results.lb_time = 0;
    results.h_time = 0;
    results.m_time = 0;

    avoc(Ws, results);

    std::cout << "\nINIT: " << init_mss << std::endl;
    std::cout << "NEW UB: " << results.heu_mss << std::endl;
    std::cout << "LB: " << results.lb_mss << std::endl;
    std::cout << "LB+: " << results.ub_mss << std::endl;
    std::cout << "ANTI OBJ: " << results.anti_obj << std::endl;
    std::cout << "SWAPS: " << results.n_swaps << std::endl;
    std::cout << "STOP: " << results.stop << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "GAP UB-LB " <<  (results.heu_mss - results.lb_mss)*100/results.heu_mss << "%" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "GAP UB-LB+ " <<  (results.heu_mss - results.ub_mss)*100/results.heu_mss << "%" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;

	test_SUMMARY
	<< results.ub_mss << "\t"
	<< results.lb_mss << "\t"
	<< results.anti_obj << "\t"
    << (results.heu_mss - results.lb_mss)*100/results.heu_mss << "%\t"
	<< (results.heu_mss - results.ub_mss)*100/results.heu_mss << "%\t"
    << results.m_time << "\t"
    << results.h_time << "\t"
    << results.n_swaps << "\t"
    << results.stop << "\t"
    << results.lb_time << "\t"
    << (results.h_time + results.m_time + results.lb_time)/60 << "\n";

    log_file.close();
	test_SUMMARY.close();

}

int main(int argc, char **argv) {

    run(argc, argv);

    return EXIT_SUCCESS;
}
