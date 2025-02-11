#ifndef ANTICLUSTERING_HEURISTICS_H
#define ANTICLUSTERING_HEURISTICS_H

#include <unordered_map>
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
#include "sdp_branch_and_bound.h"

typedef struct HResult {

	double anti_obj;
	double heu_mss;
	double lb_mss;
	double ub_mss;
	double h_time;
	double m_time;
	double lb_time;
	int n_swaps;
	int it;
	arma::mat init_sol;
	std::string stop;

} HResult;

arma::mat read_part_sol(const char *filename, int nh, double &value);
void save_to_file(arma::mat &X, std::string name);
arma::mat create_first_sol(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes);
void evaluate_swap(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, arma::mat &new_sol, std::vector<int> hList, arma::mat &new_W_hc);
void update(arma::mat &antic_sol, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes, std::vector<int> hList);
void avoc(arma::mat &Ws, HResult &results);


#endif //ANTICLUSTERING_HEURISTICS_H