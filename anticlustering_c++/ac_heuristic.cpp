#include <armadillo>
#include <iomanip>
#include "kmeans_util.h"
#include "matlab_util.h"
#include "mount_model.h"
#include "sdp_branch_and_bound.h"
#include "ThreadPoolPartition.h"
#include "ac_heuristic.h"

void save_to_file(arma::mat &X, std::string name){

    std::ofstream f;
    f.open(result_path + "_" + name + ".txt");

    for (int i = 0; i < X.n_rows; i++){
        double val = X(i,0);
        f << val;
        for (int j = 1; j < X.n_cols; j++){
            val = X(i,j);
            f << " " << val;
        }
        f << "\n";
    }
    f.close();
}

// read partial clustering sol
arma::mat read_part_sol(const char *filename, int nh, double &value) {

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

    arma::mat sol(nh, k);
    file >> value;
    for (int i = 0; i < nh; i++) {
        if (cols == 1) {
            sol.row(i) = arma::zeros(k).t();
            double c = 0;
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

void avoc(arma::mat &data, HResult &results) {

	std::cout << std::endl << "Running heuristics anticluster with k-means esteeme..\n\n";

	// Create an initial partition and save init centroids
    std::vector<std::vector<std::vector<int>>> sol_cls(k);
	arma::mat centroids_heu = arma::zeros(k, d);
	std::random_device rd;
	std::mt19937 gen(rd());
    for (int c = 0; c < k; ++c) {
        sol_cls[c] = std::vector<std::vector<int>>(p);
    	std::vector<int> all_points;
    	all_points.reserve(n);
        for (int i = 0; i < n; ++i)
    		if (results.init_sol(i,c) == 1) {
    			all_points.push_back(i);
    			centroids_heu.row(c) += data.row(i);
    		}
    	int nc = all_points.size();
    	if (nc == 0)
    		std::printf("Heuristic centroids: cluster %d is empty!\n", c);
    	centroids_heu.row(c) /= nc;

    	std::shuffle(all_points.begin(), all_points.end(), gen);
    	for (int h = 0; h < p; ++h) {
			int points = floor(nc/p);
			if (h <  nc % p)
				points += 1;
    		sol_cls[c][h] = std::vector<int>(points);
        	for (int i = 0; i < points; ++i) {
        		sol_cls[c][h][i] = all_points.back();
        		all_points.pop_back();
            }
    	}
    }

	auto start_time_m = std::chrono::high_resolution_clock::now();

	std::vector<std::vector<double>> all_dist(n, std::vector<double>(n));
	for (int i = 0; i < n; ++i) {
		for (int j = i+1; j < n; ++j) {
			double dist = std::pow(arma::norm(data.row(i).t() - data.row(j).t(), 2),2);
			all_dist[i][j] = dist;
			all_dist[j][i] = dist;
		}
	}

    // mount cluster partitions
    std::vector<std::vector<int>> init_sol(p);

    try {
    	GRBEnv *env = new GRBEnv();
    	mount_model *model = new mount_gurobi_model(env, k*(k-1)*p*p/2, all_dist, sol_cls);

    	model->add_point_constraints();
    	model->add_cls_constraints();
    	model->add_edge_constraints();
    	model->optimize();

    	init_sol = model->get_x_solution(sol_cls);

    	delete env;

    } catch (GRBException &e) {
    	std::cout << "Error code = " << e.getErrorCode() << std::endl;
    	std::cout << e.getMessage() << std::endl;
    }

	// save mount time
	auto end_time_m = std::chrono::high_resolution_clock::now();
	results.m_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_m - start_time_m).count();

    auto start_time_h = std::chrono::high_resolution_clock::now();

	// Create first sol anticluter
    arma::mat antic_sol = arma::zeros(n,p);
	for (int h = 0; h < p; ++h)
    	for (int i : init_sol[h])
    		antic_sol(i, h) = 1;

	//Create first ub sol
	std::vector<std::vector<std::vector<int>>> points(k);
	std::vector<std::vector<int>> sizes(k);
	for (int c = 0; c < k; c++) {
		points[c] = std::vector<std::vector<int>>(p);
		sizes[c] = std::vector<int>(p);
	}
	arma::mat ub_sol = arma::zeros(n,k);
	arma::mat W_hc = create_first_sol(data, antic_sol, centroids_heu, ub_sol, points, sizes);
	double best_W = arma::accu(W_hc);
	double best_GAP = (results.heu_mss - best_W)*100/results.heu_mss;

	log_file << "\n\nAVOC Algorithm\n";

	results.it = 0;
	results.n_swaps = 0;
	//Start swap heuristic
	log_file << "\n\nIt | k | LB+ | GAP %";
	log_file << "\n" << results.it << " | " << 0 << " | " << std::setprecision(2) << best_W << " | " << std::setprecision(4)  <<  best_GAP;

	std::printf("\n\nAnticlustering Heuristic\niter | k | LB+ | GAP x100");
	std::printf("\n%d | %d | %.2f | %.3f", 0, 0, best_W, best_GAP);

	while (true) {

		results.it++;

		if (best_GAP < min_gap) {
			log_file << "\n\nMin GAP " << std::setprecision(3) << min_gap << " reached\n";
			std::cout << "\n\nMin GAP " << std::setprecision(3) << min_gap << " reached\n";
			results.stop = "tol";
			break;
		}

		auto time_h = std::chrono::high_resolution_clock::now();
		if (std::chrono::duration_cast<std::chrono::seconds>(time_h - start_time_h).count() > max_time*p) {
			log_file << "\n\nTime limit reached\n";
			std::cout << "\n\nTime limit reached\n";
			results.stop = "time";
			break;
		}

		bool found_better = false;

		for (int c = 0; c < k; c++) {
			arma::uvec worst_p = arma::sort_index(W_hc.col(c), "ascend");
			for (int idx1 = 0; idx1 < p; idx1++) {

				bool found_better_ch = false;

				// Compute centroid to select nearest point
				int h1 = worst_p(idx1);
				int idx_p1 = 0;
				int p1 = points[c][h1][idx_p1];
				for (int idx2 = p-1; idx2 >= 0 && !found_better_ch; idx2--) {
					if (idx1 != idx2) {
					int h2 = worst_p(idx2);
					for (int idx_p2 = points[c][h2].size()-1; idx_p2 >= 0 && !found_better_ch; idx_p2--) {
						int p2 = points[c][h2][idx_p2];

						// evaluate swap
						arma::mat new_sol = ub_sol;
						arma::mat new_antic_sol = antic_sol;
						new_antic_sol(p1,h1)=0;
						new_antic_sol(p1,h2)=1;
						new_antic_sol(p2,h1)=1;
						new_antic_sol(p2,h2)=0;
						std::vector<int> hList;
    					hList.push_back(h1);
    					hList.push_back(h2);
						arma::mat new_W_hc = W_hc;
						evaluate_swap(data, new_antic_sol, centroids_heu, new_sol, hList, new_W_hc);
						double W = arma::accu(new_W_hc);
						if (W > best_W) {
							found_better = true;
							found_better_ch = true;
							best_W = W;
							W_hc = new_W_hc;
							ub_sol = new_sol;
							antic_sol = new_antic_sol;
							update(antic_sol, ub_sol, points, sizes, hList);
							best_GAP = (results.heu_mss - best_W)*100/ results.heu_mss;

							results.n_swaps++;
							log_file << "\n" << results.it << " | " << c+1 << " | " << std::setprecision(2) << best_W << " | " << std::setprecision(4) << best_GAP;
							std::printf("\n%d | %d | %.2f | %.3f", results.it, c+1, best_W, best_GAP);


						}
					}
				}
				}
			}
		}

		if (!found_better) {
			log_file << "\n\nNo better sol" << "\n\n";
			std::cout << "\n\nNo better sol " << "\n\n";
			results.stop = "stall";
			break;
		}

    }

	results.anti_obj = best_W;
	std::vector<std::vector<int>> sol(p);
	for (int h = 0; h < p; h++) {
		sol[h].reserve(n);
		for (int c = 0; c < k; c++)
			for (int i = 0; i < sizes[c][h]; i++) {
				sol[h].push_back(points[c][h][i]);
			}
		std::cout << "\n";
	}
	log_file << "\n---------------------------------------------------------------------\n";

	// save heuristic time
	auto end_time_h = std::chrono::high_resolution_clock::now();
	results.h_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_h - start_time_h).count();

	// create true lower bound
	auto start_time_lb = std::chrono::high_resolution_clock::now();

    auto *shared_data_part = new SharedDataPartition();

    shared_data_part->threadStates.reserve(n_threads_part);
    for (int i = 0; i < n_threads_part; i++)
        shared_data_part->threadStates.push_back(false);

    for (int h = 0; h < p; ++h) {
        auto *job = new PartitionJob();
        job->part_id = h;
		int nh = sol[h].size();

        arma::mat data_part(nh, d);
		for (int i = 0; i < nh; i++)
			data_part.row(i) = data.row(sol[h][i]);

		// Call python to compute k-means ub with improved heuristic
    	save_to_file(data_part, "pf_" + std::to_string(h));
		std::string str_path = result_path + "_pf_" + std::to_string(h) + ".txt";
        std::string out_assignment = result_path + "_acfinal_" + std::to_string(h) + ".txt";
        std::string args = str_path + " " + std::to_string(k) + " " + std::to_string(10000) + " " + out_assignment;
        std::string command = "python3 ../run_kmeans.py " + args;
        std::cout << command << "\n";
        int system_val = system(command.c_str());
        if (system_val == -1) {
            // The system method failed
            std::cout << "Failed to call Python script" << "\n";
            exit(EXIT_FAILURE);
        }

		// Get sol and objs
		arma::mat final_sol(nh,k);
		double part_value = 0;
        final_sol = read_part_sol(out_assignment.c_str(), nh, part_value);

    	job->max_ub = part_value;
        job->part_data = data_part;
        shared_data_part->queue.push_back(job);
        shared_data_part->print = true;

    	//save_to_file(data_part, "PART_" + std::to_string(h));
    }

    ThreadPoolPartition p_pool(shared_data_part, n_threads_part);
	log_file << "\n";

    while (true) {

        {
            std::unique_lock<std::mutex> l(shared_data_part->queueMutex);
            while (is_thread_pool_working(shared_data_part->threadStates)) {
                shared_data_part->mainConditionVariable.wait(l);
            }

            if (shared_data_part->queue.empty())
                break;
        }

    }

    // collect all the results
    results.lb_mss = 0;
    for (auto &lb_bound : shared_data_part->lb_part)
        results.lb_mss += lb_bound;
    results.ub_mss = 0;
    for (auto &ub_bound : shared_data_part->ub_part)
        results.ub_mss += ub_bound;

	std::map<int, arma::mat> sol_map;
    for (int h = 0; h < p; h++) {
    	arma::vec arma_sol = arma::conv_to<arma::vec>::from(sol[h]);
    	sol_map[h] = std::move(arma::join_horiz(arma_sol, shared_data_part->sol_part[h]));
    }

    p_pool.quitPool();

    // free memory
    delete (shared_data_part);

	auto end_time_lb = std::chrono::high_resolution_clock::now();
	results.lb_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_lb - start_time_lb).count();

	std::cout << std::endl << "*********************************************************************" << std::endl;

	log_file << std::endl << "*********************************************************************" << std::endl;
    log_file << "LB: " << results.lb_mss;
    log_file << "\nLB+: " << results.ub_mss;
    log_file << "\nANTI OBJ: " << results.anti_obj;
	log_file << "\nInit sol: " << results.heu_mss;
	log_file << "\nSWAPS: " << results.n_swaps;
	log_file << "\nSTOP: " << results.stop;
	log_file << "\nGAP UB-LB " <<  (results.heu_mss - results.lb_mss)*100/results.heu_mss;
	log_file << "\nGAP UB-LB+ " <<  (results.heu_mss - results.ub_mss)*100/results.heu_mss;
	log_file << std::endl << "*********************************************************************" << std::endl;
	log_file << "\n\n";

}

arma::mat create_first_sol(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes) {

	arma::mat obj(p,k);
	for (int h = 0; h < p; h++) {

		int nh = arma::accu(antic_sol.col(h));
		arma::mat data_antic(nh,d);
		arma::vec points(nh);
		nh = 0;
		for (int i = 0; i < n; ++i) {
			if (antic_sol(i,h) == 1) {
				data_antic.row(nh) = data.row(i);
				points(nh) = i;
				nh++;
			}
		}

		// Call python to compute first k-means with improved heuristic
    	save_to_file(data_antic, "pi_" + std::to_string(h));
		std::string str_path = result_path + "_pi_" + std::to_string(h) + ".txt";
        std::string out_assignment = result_path + "_acinit_" + std::to_string(h) + ".txt";
        std::string args = str_path + " " + std::to_string(k) + " " + std::to_string(10000) + " " + out_assignment;
        std::string command = "python3 ../run_kmeans.py " + args;
        std::cout << command << "\n";
        int system_val = system(command.c_str());
        if (system_val == -1) {
            // The system method failed
            std::cout << "Failed to call Python script" << "\n";
            exit(EXIT_FAILURE);
        }

		// Get sol and objs
		arma::mat sol(nh,k);
		double part_value = 0;
        sol = read_part_sol(out_assignment.c_str(), nh, part_value);
        arma::mat centroids = arma::zeros(k, d);
    	arma::vec count = arma::zeros(k);
    	for (int i = 0; i < nh; i++) {
			for (int c = 0; c < k; c++) {
				if (sol(i,c)==1) {
					centroids.row(c) += data_antic.row(i);
					count(c) += 1;
				}
    		}
    	}

		for (int c = 0; c < k; c++) {
       		if (count(c) == 0)
            	std::printf("computeCentroids(): cluster %d is empty!\n", c);
        	centroids.row(c) /= count(c);
    	}

        arma::mat m = data_antic - sol * centroids;
		arma::vec obj_h(k);
		for (int c = 0; c < k; c++) {
			arma::vec diff_squared = arma::sum((sol.col(c).t()) * arma::square(m), 1);
			obj_h(c) = arma::accu(diff_squared);
		}

		// map to the right centroids
		std::vector<int> c_idx(k);
		for (int i = 0; i < k; ++i)
			c_idx[i] = i;
		std::vector<int> mapping(k, -1);
		for (int c1 = 0; c1 < k; ++c1) {
			double min_distance = std::numeric_limits<double>::max();
			int best_match = -1;
			int best_idx = -1;
			for (int idx = 0; idx < c_idx.size(); idx++) {
				// Calculate the Euclidean distance between centroids
				int c2 = c_idx[idx];
				double distance = std::pow(arma::norm(centroids_heu.row(c2).t() - centroids.row(c1).t(), 2), 2);
				if (distance < min_distance) {
					min_distance = distance;
					best_match = c2;
					best_idx = idx;
				}
			}
			mapping[c1] = best_match;
			c_idx.erase(c_idx.begin() + best_idx);
		}

		for (int c = 0; c < k; ++c) {
			for (int i = 0; i < nh; i++)
				if (sol(i,c)==1)
					ub_sol(points(i), mapping[c]) = 1;
			obj(h, c) = obj_h(mapping[c]);
		}
    }

	//Save new sol as points and sizes
	for (int h = 0; h < p; ++h) {
		for (int c = 0; c < k; c++) {
			points[c][h].reserve(n);
			int nc = 0;
			for (int i = 0; i < n; i++)
				if (antic_sol(i,h)==1 and ub_sol(i,c)==1) {
					points[c][h].push_back(i);
					nc++;
				}
			sizes[c][h] = nc;
		}
	}

    return obj;

}


void evaluate_swap(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, arma::mat &new_sol, std::vector<int> hList, arma::mat &new_W_hc) {

	for (int h : hList) {

		int nh = arma::accu(antic_sol.col(h));
		arma::mat data_antic(nh,d);
		arma::vec points(nh);
		nh = 0;
		for (int i = 0; i < n; ++i) {
			if (antic_sol(i,h) == 1) {
				data_antic.row(nh) = data.row(i);
				points(nh) = i;
				nh++;
			}
		}

		Kmeans kmeans(data_antic, k, kmeans_verbose);
		kmeans.start(kmeans_max_it, 1, centroids_heu);
		new_W_hc.row(h) = kmeans.objectiveFunctionCls().t();

		arma::mat sol(nh,k);
		sol = kmeans.getAssignments();
		for (int i = 0; i < nh; ++i) {
			for (int c = 0; c < k; ++c) {
				if (sol(i,c)==1)
					new_sol(points(i), c) = 1;
				else
					new_sol(points(i), c) = 0;
			}
		}

	}

}

void update(arma::mat &antic_sol, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes, std::vector<int> hList) {

	//Save new sol as points and sizes
	for (int c = 0; c < k; c++) {
		for (int h : hList) {
			points[c][h].clear();
			points[c][h].reserve(n);
			int nc = 0;
			for (int i = 0; i < n; i++)
				if (antic_sol(i,h)==1 and ub_sol(i,c)==1) {
					points[c][h].push_back(i);
					nc++;
				}
			sizes[c][h] = nc;
		}
	}

}