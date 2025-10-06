#ifndef CLUSTERING_MOUNT_MODEL_H
#define CLUSTERING_MOUNT_MODEL_H

#include <armadillo>
#include <gurobi_c++.h>
#include "config_params.h"
#include "ac_heuristic.h"

template<class T>
class Matrix {
public:
    int rows;
    int cols;
private:
    std::vector<T> data;
public:
    Matrix(int row, int col);
    T&        operator()(size_t row, size_t col);
    const T&  operator()(size_t row, size_t col) const;
};

class mount_model {

	protected:
	int status;
	int m;
    std::vector<std::vector<std::vector<int>>> sol_cls;

	std::string get_x_variable_name(int c1, int h1, int t);
	std::string get_y_variable_name(int c1, int h1, int c2, int h2, int t);

	public:
	virtual void reset() = 0;
	virtual void add_point_constraints() = 0;
	virtual void add_cls_constraints() = 0;
	virtual void add_edge_constraints() = 0;
	virtual void update_Y_objective(const SymmDist& dist, const std::vector<std::vector<std::vector<int>>>& sol_cls) = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual std::vector<std::vector<int>> get_x_solution(std::vector<std::vector<std::vector<int>>> &sol_cls) = 0;
};

class mount_gurobi_model : public mount_model {

private:
	GRBEnv *env;
	GRBModel model;
	Matrix<GRBVar> X;
	Matrix<GRBVar> Y;

	Matrix<GRBVar> create_X_variables(GRBModel &model);
	Matrix<GRBVar> create_Y_variables(GRBModel &model);

public:
	mount_gurobi_model(GRBEnv *env, int m);
	virtual void add_point_constraints() override;
	virtual void add_cls_constraints() override;
	virtual void add_edge_constraints() override;
	virtual void update_Y_objective(const SymmDist& dist, const std::vector<std::vector<std::vector<int>>>& sol_cls) override;
	void reset() override;
	void optimize() override;
	virtual double get_value() override;
	virtual std::vector<std::vector<int>> get_x_solution(std::vector<std::vector<std::vector<int>>> &sol_cls) override;
};

#endif //CLUSTERING_MOUNT_MODEL_H
