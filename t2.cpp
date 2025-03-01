#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <limits>

// ====================== Sequence Class ======================
class Sequence {
private:
    std::string id;
    std::string data;

public:
    Sequence() {}
    Sequence(const std::string& id, const std::string& data) : id(id), data(data) {}

    const std::string& getId() const { return id; }
    const std::string& getData() const { return data; }
    size_t length() const { return data.length(); }

    static std::vector<Sequence> readFasta(const std::string& filename) {
        std::vector<Sequence> sequences;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        std::string line, id, data;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            if (line[0] == '>') {
                if (!id.empty()) {
                    sequences.emplace_back(id, data);
                    data.clear();
                }
                id = line.substr(1);
            } else {
                data += line;
            }
        }
        
        // Add the last sequence
        if (!id.empty()) {
            sequences.emplace_back(id, data);
        }
        
        file.close();
        return sequences;
    }
};

// ====================== Alignment Class ======================
class Alignment {
public:
    // Needleman-Wunsch algorithm for global alignment
    static double needlemanWunsch(const Sequence& seq1, const Sequence& seq2, 
                                  double match = 1.0, double mismatch = -1.0, double gap = -2.0) {
        const std::string& s1 = seq1.getData();
        const std::string& s2 = seq2.getData();
        size_t n = s1.length();
        size_t m = s2.length();
        
        // Create scoring matrix
        std::vector<std::vector<double>> H(n + 1, std::vector<double>(m + 1, 0.0));
        
        // Initialize first row and column with gap penalties
        for (size_t i = 0; i <= n; i++) {
            H[i][0] = i * gap;
        }
        
        for (size_t j = 0; j <= m; j++) {
            H[0][j] = j * gap;
        }
        
        // Fill the matrix
        for (size_t i = 1; i <= n; i++) {
            for (size_t j = 1; j <= m; j++) {
                double diagonal = H[i-1][j-1] + (s1[i-1] == s2[j-1] ? match : mismatch);
                double up = H[i-1][j] + gap;
                double left = H[i][j-1] + gap;
                
                H[i][j] = std::max({diagonal, up, left});
            }
        }
        
        // Return the alignment score
        return H[n][m];
    }
    
    // Compute distance from alignment score
    static double computeDistance(double alignmentScore, const Sequence& seq1, const Sequence& seq2) {
        // Normalize the alignment score by the sequence lengths to get a distance measure
        // Higher scores mean better alignment (lower distance)
        double maxPossibleScore = std::min(seq1.length(), seq2.length());
        
        // Convert to distance: lower score means higher distance
        if (maxPossibleScore == 0) return 0.0;
        
        // Normalize to [0,1] range and invert (1 - normalized_score)
        double normalizedScore = alignmentScore / maxPossibleScore;
        return 1.0 - normalizedScore;
    }
};

// ====================== Distance Matrix Class ======================
class DistanceMatrix {
private:
    std::vector<std::vector<double>> matrix;
    size_t size;

public:
    DistanceMatrix(size_t n) : size(n) {
        matrix.resize(n, std::vector<double>(n, 0.0));
    }

    void set(size_t i, size_t j, double value) {
        matrix[i][j] = value;
        matrix[j][i] = value;  // Ensure symmetry
    }

    double get(size_t i, size_t j) const {
        return matrix[i][j];
    }

    size_t getSize() const {
        return size;
    }

    void print() const {
        std::cout << "\nDistance Matrix:" << std::endl;
        // Print header
        std::cout << "      ";
        for (size_t j = 0; j < size; j++) {
            std::cout << "Seq" << std::setw(5) << j+1 << " ";
        }
        std::cout << std::endl;
        
        for (size_t i = 0; i < size; ++i) {
            std::cout << "Seq" << std::setw(2) << i+1 << " ";
            for (size_t j = 0; j < size; ++j) {
                std::cout << std::fixed << std::setprecision(4) << std::setw(7) << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Calculate the number of total pairs for n sequences
    static size_t calculateTotalPairs(size_t n) {
        return (n * (n - 1)) / 2;
    }

    // Convert pair index to matrix indices
    static void pairIndexToMatrixIndices(size_t pairIndex, size_t n, size_t& i, size_t& j) {
        // Convert a flat pairIndex to matrix coordinates (i,j) where i < j
        // This is based on the triangular indexing formula
        i = n - 2 - floor(sqrt(-8 * pairIndex + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
        j = pairIndex + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
    }
};

// ====================== Guide Tree Class ======================
struct Node {
    int id;
    Node* left;
    Node* right;
    double height;
    
    Node(int id) : id(id), left(nullptr), right(nullptr), height(0.0) {}
    Node(Node* left, Node* right, double height) 
        : id(-1), left(left), right(right), height(height) {}
    
    bool isLeaf() const { return left == nullptr && right == nullptr; }
    
    ~Node() {
        delete left;
        delete right;
    }
};

class GuideTree {
private:
    Node* root;

public:
    GuideTree() : root(nullptr) {}
    
    // UPGMA algorithm to build the guide tree
    void buildUPGMA(const DistanceMatrix& distMatrix) {
        size_t n = distMatrix.getSize();
        
        // Initialize a vector of leaf nodes
        std::vector<Node*> nodes;
        for (size_t i = 0; i < n; i++) {
            nodes.push_back(new Node(i));
        }
        
        // Create a mutable copy of the distance matrix
        std::vector<std::vector<double>> distances(n, std::vector<double>(n));
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                distances[i][j] = distMatrix.get(i, j);
            }
        }
        
        // Keep track of node sizes (initially 1 for all leaf nodes)
        std::vector<int> nodeSizes(n, 1);
        
        // UPGMA algorithm
        for (size_t iter = 0; iter < n - 1; iter++) {
            // Find the pair with minimum distance
            double minDist = std::numeric_limits<double>::max();
            size_t minI = 0, minJ = 0;
            
            for (size_t i = 0; i < nodes.size(); i++) {
                for (size_t j = i + 1; j < nodes.size(); j++) {
                    if (distances[i][j] < minDist) {
                        minDist = distances[i][j];
                        minI = i;
                        minJ = j;
                    }
                }
            }
            
            // Create a new internal node
            Node* newNode = new Node(nodes[minI], nodes[minJ], minDist / 2.0);
            
            // Update distances
            size_t newIndex = nodes.size();
            distances.push_back(std::vector<double>(newIndex + 1));
            
            int newNodeSize = nodeSizes[minI] + nodeSizes[minJ];
            nodeSizes.push_back(newNodeSize);
            
            for (size_t k = 0; k < newIndex; k++) {
                if (k != minI && k != minJ) {
                    // Weighted average of distances
                    double newDist = (distances[minI][k] * nodeSizes[minI] + 
                                    distances[minJ][k] * nodeSizes[minJ]) / newNodeSize;
                    
                    distances[newIndex][k] = newDist;
                    distances[k][newIndex] = newDist;
                }
            }
            
            // Remove the two joined nodes (in reverse order to maintain indices)
            if (minJ > minI) {
                nodes.erase(nodes.begin() + minJ);
                nodes.erase(nodes.begin() + minI);
            } else {
                nodes.erase(nodes.begin() + minI);
                nodes.erase(nodes.begin() + minJ);
            }
            
            // Add the new node
            nodes.push_back(newNode);
        }
        
        // The last remaining node is the root
        root = nodes[0];
    }
    
    // Print the guide tree in Newick format
    std::string getNewickFormat() const {
        if (!root) return "()";
        return newickFormat(root) + ";";
    }
    
    ~GuideTree() {
        delete root;
    }

private:
    std::string newickFormat(Node* node) const {
        if (!node) return "";
        
        if (node->isLeaf()) {
            return "Seq" + std::to_string(node->id + 1); // 1-indexed for display
        } else {
            std::stringstream ss;
            ss << "(" << newickFormat(node->left) << "," 
               << newickFormat(node->right) << ")";
            return ss.str();
        }
    }
};

// ====================== MPI Functions ======================
void computeDistanceMatrix(const std::vector<Sequence>& sequences, int rank, int numProcs) {
    size_t n = sequences.size();
    size_t totalPairs = DistanceMatrix::calculateTotalPairs(n);
    
    // Create a distance matrix (each process will compute part of it)
    DistanceMatrix distMatrix(n);
    
    // Each process computes a subset of pairwise distances
    size_t pairsPerProcess = totalPairs / numProcs;
    size_t remainder = totalPairs % numProcs;
    
    // Calculate start and end pair indices for this process
    size_t startPair = static_cast<size_t>(rank) * pairsPerProcess + 
                      (static_cast<size_t>(rank) < remainder ? rank : remainder);
    
    size_t endPair = startPair + pairsPerProcess + 
                    (static_cast<size_t>(rank) < remainder ? 1 : 0);
    
    if (rank == 0) {
        std::cout << "Total pairs to compute: " << totalPairs << std::endl;
        std::cout << "Computing pairwise alignments..." << std::endl;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Compute assigned pairwise alignments
    for (size_t pairIndex = startPair; pairIndex < endPair; pairIndex++) {
        size_t i, j;
        DistanceMatrix::pairIndexToMatrixIndices(pairIndex, n, i, j);
        
        // Only compute distances for pairs where i < j
        if (i < j) {
            double alignmentScore = Alignment::needlemanWunsch(sequences[i], sequences[j]);
            double distance = Alignment::computeDistance(alignmentScore, sequences[i], sequences[j]);
            
            // Store in local matrix part
            distMatrix.set(i, j, distance);
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    
    // Only rank 0 prints timing information
    if (rank == 0) {
        std::cout << "Process " << rank << " computed " << (endPair - startPair) 
                  << " pairs in " << elapsedTime.count() << " seconds" << std::endl;
    }
    
    // Gather all parts of the distance matrix to rank 0
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            double localValue = distMatrix.get(i, j);
            double globalValue;
            
            // Use MPI_Reduce to get the sum of all parts (only one process will have a non-zero value)
            MPI_Reduce(&localValue, &globalValue, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            
            if (rank == 0) {
                distMatrix.set(i, j, globalValue);
            }
        }
    }
    
    // Process 0 prints the final distance matrix
    if (rank == 0) {
        std::cout << "\nFinal Distance Matrix:" << std::endl;
        distMatrix.print();
        
        // Build and display the guide tree
        std::cout << "\nConstructing guide tree using UPGMA..." << std::endl;
        GuideTree tree;
        tree.buildUPGMA(distMatrix);
        std::cout << "Guide tree (Newick format): " << tree.getNewickFormat() << std::endl;
        
        // Print all sequences at the end
        std::cout << "\nInput Sequences:" << std::endl;
        for (size_t i = 0; i < sequences.size(); i++) {
            std::cout << ">Seq" << (i+1) << " " << sequences[i].getId() << std::endl;
            // Print sequence data in lines of 60 characters (standard FASTA format)
            const std::string& seqData = sequences[i].getData();
            for (size_t j = 0; j < seqData.length(); j += 60) {
                std::cout << seqData.substr(j, 60) << std::endl;
            }
        }
        
        // Print user information
        std::cout << "\nAnalysis completed by: Mustafaiqbal2" << std::endl;
        std::cout << "Date: 2025-03-01 10:01:12 UTC" << std::endl;
    }
    // Print the sequences in CLUSTAL-like format
    if (rank == 0) {
        std::cout << "\nCLUSTAL format alignment by MPI-MSA" << std::endl;
        std::cout << "\n\n";
        
        // Get the maximum sequence ID length for proper spacing
        size_t maxIdLength = 0;
        for (size_t i = 0; i < sequences.size(); i++) {
            std::string seqId = "Seq" + std::to_string(i+1);
            maxIdLength = std::max(maxIdLength, seqId.length());
        }
        
        // Print each sequence
        for (size_t i = 0; i < sequences.size(); i++) {
            std::string seqId = "Seq" + std::to_string(i+1);
            std::cout << std::left << std::setw(maxIdLength + 8) << seqId;
            std::cout << sequences[i].getData() << std::endl;
        }
        
        std::cout << "\nNote: This shows original sequences without alignment gaps." << std::endl;
        std::cout << "To perform full MSA with gap insertion, progressive alignment is required." << std::endl;
        
        // Print user information
        std::cout << "\nAnalysis completed by: " << "Mustafaiqbal2" << std::endl;
        std::cout << "Date: " << "2025-03-01 10:03:00 UTC" << std::endl;
}
}

// ====================== Main Function ======================
int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    // Check command line arguments
    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <fasta_file>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    std::string filename = argv[1];
    std::vector<Sequence> sequences;
    
    // Only rank 0 reads the file and broadcasts to all processes
    if (rank == 0) {
        try {
            std::cout << "Reading sequences from " << filename << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            sequences = Sequence::readFasta(filename);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            
            std::cout << "Read " << sequences.size() << " sequences in " 
                      << elapsed.count() << " seconds" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    // Broadcast the number of sequences
    int numSequences = sequences.size();
    MPI_Bcast(&numSequences, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Prepare to broadcast sequences
    if (rank != 0) {
        sequences.resize(numSequences);
    }
    
    // Broadcast each sequence
    for (int i = 0; i < numSequences; i++) {
        if (rank == 0) {
            // Pack sequence data
            std::string id = sequences[i].getId();
            std::string data = sequences[i].getData();
            
            // Send sizes first
            int idSize = id.size();
            int dataSize = data.size();
            MPI_Bcast(&idSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Send actual data
            MPI_Bcast(const_cast<char*>(id.c_str()), idSize, MPI_CHAR, 0, MPI_COMM_WORLD);
            MPI_Bcast(const_cast<char*>(data.c_str()), dataSize, MPI_CHAR, 0, MPI_COMM_WORLD);
        } else {
            // Receive sizes
            int idSize, dataSize;
            MPI_Bcast(&idSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Prepare buffers
            std::string id(idSize, ' ');
            std::string data(dataSize, ' ');
            
            // Receive data
            MPI_Bcast(const_cast<char*>(id.c_str()), idSize, MPI_CHAR, 0, MPI_COMM_WORLD);
            MPI_Bcast(const_cast<char*>(data.c_str()), dataSize, MPI_CHAR, 0, MPI_COMM_WORLD);
            
            // Create the sequence
            sequences[i] = Sequence(id, data);
        }
    }
    
    // Compute distance matrix in parallel
    computeDistanceMatrix(sequences, rank, numProcs);
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}