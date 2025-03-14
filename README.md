# DNA-Matching-MPI

A high-performance DNA sequence matching implementation using MPI (Message Passing Interface) for parallel processing.

## Overview

This project implements parallel DNA sequence matching algorithms using C++ and MPI to distribute the computational workload across multiple processors. By leveraging parallel computing techniques, the system can efficiently process and match large DNA sequences that would be computationally intensive on a single machine.

## Features

- Parallel DNA sequence matching using MPI
- Support for various DNA comparison algorithms
- Efficient handling of large genomic datasets
- Scalable performance across multiple nodes/processors
- Optimized C++ implementation for maximum performance

## Requirements

- C++ compiler (g++ recommended)
- MPI implementation (OpenMPI or MPICH)
- CMake (optional, for build management)

## Installation

### Setting up MPI

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install mpich libmpich-dev
```

#### On CentOS/RHEL:
```bash
sudo yum install mpich mpich-devel
```

#### On macOS (using Homebrew):
```bash
brew install open-mpi
```

### Building the Project

1. Clone the repository:
```bash
git clone https://github.com/Mustafaiqbal2/DNA-Matching-MPI.git
cd DNA-Matching-MPI
```

2. Compile the source code:
```bash
mpic++ -o dna_matching src/main.cpp src/dna_matcher.cpp -O3
```

Or if using CMake:
```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### Running the Program

Basic usage:
```bash
mpirun -np <number_of_processes> ./dna_matching <reference_sequence_file> <query_sequence_file>
```

Example with 4 processes:
```bash
mpirun -np 4 ./dna_matching data/reference.fasta data/query.fasta
```

### Input File Format

The program accepts DNA sequences in FASTA format or plain text files where each line represents a different DNA sequence.

### Output

The program will output matching results including:
- Match positions
- Similarity scores
- Execution time statistics

## Implementation Details

The implementation distributes DNA sequence matching work across multiple processes:

1. The master process (rank 0) reads and partitions the DNA data
2. Work is distributed among worker processes
3. Each process executes matching algorithms on its assigned portion
4. Results are gathered back to the master process and consolidated
5. Final results are output to the user

## Performance Considerations

- Performance scales with the number of processors, but communication overhead may limit efficiency at very high process counts
- Optimal performance typically achieved when data chunks are large enough to justify parallel processing
- Memory usage increases with sequence length and number of processes

## Contributing

Contributions to improve the project are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- [Mustafa Iqbal](https://github.com/Mustafaiqbal2)

## Acknowledgments

- This project was developed as part of [mention any course, research project, etc. if applicable]
- Thanks to [mention any contributors, libraries, or resources that were helpful]

## Last Updated
2025-03-14
