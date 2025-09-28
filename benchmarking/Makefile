CC = clang
CXX = clang++

CFLAGS = -O3 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
CXXFLAGS = -O3 -std=c++11 -Xpreprocessor -fopenmp \
           -I/opt/homebrew/opt/libomp/include \
           -Ifaiss

FAISS_DIR = faiss/faiss
FAISS_LIB_DIR = faiss/build/faiss
FAISS_INCLUDE = build/faiss
LIBOMP_DIR = /opt/homebrew/opt/libomp/lib
LDFLAGS = -L$(FAISS_LIB_DIR) -L$(LIBOMP_DIR) -lfaiss -lomp -lm -lstdc++ -framework Accelerate
LDFLAGS += -L$(FAISS_LIB_DIR) -lfaiss

WRAPPER_SRC = faiss_wrapper.cpp
WRAPPER_OBJ = faiss_wrapper.o

OBJS = benchmark_main.o benchmark_runner.o benchmark_utils.o $(WRAPPER_OBJ)

TARGET = benchmark

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

benchmark_main.o: benchmark_main.c
	$(CC) $(CFLAGS) -c $<

benchmark_runner.o: benchmark_runner.c
	$(CC) $(CFLAGS) -c $<

benchmark_utils.o: benchmark_utils.c
	$(CC) $(CFLAGS) -c $<

faiss_wrapper.o: $(WRAPPER_SRC)
	$(CXX) $(CXXFLAGS) -I$(FAISS_DIR) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
