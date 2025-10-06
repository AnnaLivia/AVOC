#include "mount_model.h"

template<class T>
Matrix<T>::Matrix(int row, int col) : rows(row), cols(col), data(rows*cols) {}

template<class T>
T& Matrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols + col];
}

template<class T>
const T& Matrix<T>::operator()(size_t row, size_t col) const {
    return data[row*cols + col];
}

std::string mount_model::get_x_variable_name(int c1, int h1, int t){
	std::ostringstream os;
	os << "X" << c1 << "_" << h1 << "_" << t;
	return os.str();
}

std::string mount_model::get_y_variable_name(int c1, int h1, int c2, int h2, int t){
	std::ostringstream os;
	os << "Y" << c1 << "_" << h1 << "_" << c2 << "_" << h2 << "_" << t;
	return os.str();
}

// Constructor for mount_gurobi_model
mount_gurobi_model::mount_gurobi_model(GRBEnv* env, int m)
    : model(*env), X(k * p, p), Y(m, p)
{
    this->m       = m;
    this->env     = env;

    this->X = create_X_variables(this->model);
    this->Y = create_Y_variables(this->model);

    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_DoubleParam_TimeLimit, 60.0);
    model.set(GRB_DoubleParam_MIPGap, 0.05);
}

// Create X variables in the model
Matrix<GRBVar> mount_gurobi_model::create_X_variables(GRBModel& model) {
    Matrix<GRBVar> Xvars(k * p, p);
    for (int c = 0; c < k; ++c) {
        for (int h = 0; h < p; ++h) {
            const int s = c * p + h; // row in X
            for (int t = 0; t < p; ++t) {
                Xvars(s, t) = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "");
            }
        }
    }
    return Xvars;
}

// Create Y variables in the model
Matrix<GRBVar> mount_gurobi_model::create_Y_variables(GRBModel& model) {
    Matrix<GRBVar> Yvars(m, p);
    int s = 0;
    for (int c1 = 0; c1 < k - 1; ++c1)
        for (int h1 = 0; h1 < p; ++h1)
            for (int c2 = c1 + 1; c2 < k; ++c2)
                for (int h2 = h1; h2 < p; ++h2, ++s)
                    for (int t = 0; t < p; ++t)
                        Yvars(s, t) = model.addVar(0.0, 1.0, /*obj=*/0.0, GRB_BINARY);
    return Yvars;
}

void mount_gurobi_model::update_Y_objective(const SymmDist& dist,
                                            const std::vector<std::vector<std::vector<int>>>& sol_cls)
{
    int s = 0;
    arma::vec rA(n, arma::fill::zeros);

    for (int c1 = 0; c1 < k - 1; ++c1) {
        for (int h1 = 0; h1 < p; ++h1) {
            rA.zeros();
            for (int i : sol_cls[c1][h1]) {
                for (int j = 0; j < i; ++j)     rA[j] += dist.at(j, i);
                for (int j = i + 1; j < n; ++j) rA[j] += dist.at(i, j);
            }

            for (int c2 = c1 + 1; c2 < k; ++c2) {
                for (int h2 = h1; h2 < p; ++h2) {
                    double obj = 0.0;
                    for (int j : sol_cls[c2][h2]) obj += rA[(size_t)j];

                    for (int t = 0; t < p; ++t) {
                        Y(s, t).set(GRB_DoubleAttr_Obj, -obj);
                    }
                    ++s;
                }
            }
        }
    }
}



void mount_gurobi_model::add_point_constraints() {
    for (int c = 0; c < k; ++c) {
        for (int h = 0; h < p; ++h) {
            const int s = c * p + h;
            GRBLinExpr sum = 0;
            for (int t = 0; t < p; ++t) sum += X(s, t);
            model.addConstr(sum == 1);
        }
    }
}

void mount_gurobi_model::add_cls_constraints() {
    for (int t = 0; t < p; ++t) {
        for (int c = 0; c < k; ++c) {
            GRBLinExpr sum = 0;
            for (int h = 0; h < p; ++h) {
                const int s = c * p + h;
                sum += X(s, t);
            }
            model.addConstr(sum == 1);
        }
    }
}

void mount_gurobi_model::add_edge_constraints() {
    for (int t = 0; t < p; ++t) {
        int s = 0; // iterate Y rows in the same order used when creating Y
        for (int c1 = 0; c1 < k - 1; ++c1) {
            for (int h1 = 0; h1 < p; ++h1) {
                for (int c2 = c1 + 1; c2 < k; ++c2) {
                    for (int h2 = h1; h2 < p; ++h2) {
                        const int s1 = c1 * p + h1;   // row of X for (c1,h1)
                        const int s2 = c2 * p + h2;   // row of X for (c2,h2)
                        model.addConstr(Y(s, t) <= X(s1, t));
                        model.addConstr(Y(s, t) <= X(s2, t));
                        model.addConstr(Y(s, t) >= X(s1, t) + X(s2, t) - 1);
                        ++s;
                    }
                }
            }
        }
    }
}

void mount_gurobi_model::optimize(){
	try {
        //std::string file = sol_path;
        //auto name = file.substr(0, file.find_last_of("."));
        //model.write(name + ".lp");
		model.optimize();
		status = model.get(GRB_IntAttr_Status);
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}

void mount_gurobi_model::reset() {
    model.update();
    model.reset();   // clears previous B&B state, keeps vars/cons intact
}

std::vector<std::vector<int>>
mount_gurobi_model::get_x_solution(std::vector<std::vector<std::vector<int>>>& sol_cls) {
    std::vector<std::vector<int>> opt(p);
    for (int t = 0; t < p; ++t) opt[t].reserve(n);

    for (int c = 0; c < k; ++c) {
        for (int h = 0; h < p; ++h) {
            const int s = c * p + h;

            int chosen_t = -1;
            for (int t = 0; t < p; ++t) {
                if (X(s, t).get(GRB_DoubleAttr_X) > 0.8) { chosen_t = t; break; }
            }

            // append points in (c,h) to partition chosen_t
            for (int i : sol_cls[c][h]) opt[chosen_t].push_back(i);
        }
    }
    return opt;
}

double mount_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}


