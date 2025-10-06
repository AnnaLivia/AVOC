# ===== Paths =====
matlab_path = /home/aelle
gurobi_path = /home/aelle/gurobi1100/linux64

# ===== Sources / Objects =====
sources = main.cpp ac_heuristic.cpp mount_model.cpp sdp_solver_util.cpp sdp_branch_and_bound.cpp \
          matlab_util.cpp kmeans_util.cpp Kmeans.cpp JobQueue.cpp util.cpp ilp_model.cpp \
          feasibility.cpp ub_heuristics.cpp ThreadPool.cpp ThreadPoolPartition.cpp
objects = $(sources:.cpp=.o)


# ===== Targets =====
all: avoc

avoc: $(objects)
	g++ -L $(matlab_path)/extern/bin/glnxa64/ \
		-L $(gurobi_path)/lib/ \
		-o $@ $^ -larmadillo -pthread -llapack -lopenblas -lMatlabDataArray -lMatlabEngine -lgurobi_c++ -lgurobi110 \
		-Wl,-rpath,$(matlab_path)/extern/bin/glnxa64/  \
		-Wl,-rpath,$(gurobi_path)/lib

%.o: %.cpp
		g++ -O3 -march=native -ffast-math -DNDEBUG -DARMA_NO_DEBUG  \
		-o $@ -c $< -I $(matlab_path)/extern/include/ -I $(gurobi_path)/include

.PHONY: clean		
clean:
		rm -f avoc *.o
