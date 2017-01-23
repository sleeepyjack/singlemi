all:
	bash -c "git submodule init;"
	bash -c "git submodule update;"
	nvcc -O3 -std=c++11 -arch=sm_52 -w main.cu snpreader/SNPReader.cpp snpreader/LineReader.cpp -o singlemi
