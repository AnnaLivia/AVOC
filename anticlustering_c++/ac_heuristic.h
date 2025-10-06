#ifndef ANTICLUSTERING_HEURISTICS_H
#define ANTICLUSTERING_HEURISTICS_H

#include "sdp_branch_and_bound.h"

typedef struct HResult
{
	double anti_obj;
	double first_mss;
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

struct SymmDist
{
	int n{};
	std::vector<float> data; // stores i<j upper triangle

	explicit SymmDist(int n_) : n(n_), data(size_t(n_) * (n_ - 1) / 2)
	{
	}

	// map (i,j) with i<j to linear index
	inline size_t idx(int i, int j) const
	{
		// ensure i < j
		if (i > j) std::swap(i, j);
		// skip diagonal
		// index formula for upper-triangular packed storage (row-major, excluding diagonal)
		return size_t(i) * (2 * n - i - 1) / 2 + (j - i - 1);
	}

	inline float at(int i, int j) const
	{
		return (i == j) ? 0.0f : data[idx(i, j)];
	}

	inline float& at(int i, int j)
	{
		if (i == j)
		{
			static float zero = 0.0f;
			return zero;
		} // never written
		return data[idx(i, j)];
	};
};

void save_to_file(arma::mat& X, std::string name);
void avoc(arma::mat& Ws, HResult& results);


arma::mat create_first_sol(arma::mat& data,
						   arma::mat& antic_sol,
						   arma::mat& centroids_heu,
						   arma::mat& ub_sol,
						   std::vector<std::vector<std::vector<int>>>& points,
						   std::vector<std::vector<int>>& sizes);
std::pair<double, double> compute_lb(arma::mat& data, const std::vector<std::vector<int>>& sol, bool verbose);

#endif //ANTICLUSTERING_HEURISTICS_H
