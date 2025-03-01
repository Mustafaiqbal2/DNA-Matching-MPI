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
#include <cstdio>
#include <map>
#include <unordered_map>
#include <stack>

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

    void print(std::ofstream& outFile) const {
        outFile << "Distance Matrix:" << std::endl;
        // Print header
        outFile << "      ";
        for (size_t j = 0; j < size; j++) {
            outFile << "Seq" << std::setw(5) << j+1 << " ";
        }
        outFile << std::endl;
        
        for (size_t i = 0; i < size; ++i) {
            outFile << "Seq" << std::setw(2) << i+1 << " ";
            for (size_t j = 0; j < size; ++j) {
                outFile << std::fixed << std::setprecision(4) << std::setw(7) << matrix[i][j] << " ";
            }
            outFile << std::endl;
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
    
    // Get the root node for traversal
    Node* getRoot() const {
        return root;
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

// ====================== Progressive Alignment Class ======================
class ProgressiveAlignment {
public:
    struct AlignedSequence {
        std::string id;
        std::string originalData;
        std::string alignedData;  // With gaps inserted
        
        AlignedSequence(const Sequence& seq) : 
            id(seq.getId()), originalData(seq.getData()), alignedData(seq.getData()) {}
        
        AlignedSequence(const std::string& id, const std::string& original, const std::string& aligned) :
            id(id), originalData(original), alignedData(aligned) {}
        
        size_t length() const { return alignedData.length(); }
    };
    
    using Profile = std::vector<AlignedSequence>;
    
    // Perform the progressive alignment based on the guide tree
    static Profile alignProgressive(const std::vector<Sequence>& sequences, const GuideTree& tree) {
        Node* root = tree.getRoot();
        if (!root) {
            return {}; // Empty profile if no tree
        }
        
        // Map to store intermediate alignments at each node
        std::unordered_map<Node*, Profile> nodeProfiles;
        
        // Perform post-order traversal to build alignments from leaves to root
        traverseAndAlign(root, sequences, nodeProfiles);
        
        // Return the profile at the root
        return nodeProfiles[root];
    }
    
    // Generate conservation symbols for CLUSTAL format
    static std::string generateConservationLine(const Profile& profile) {
        if (profile.empty()) return "";
        
        size_t length = profile[0].alignedData.length();
        std::string conservationLine(length, ' ');
        
        for (size_t i = 0; i < length; i++) {
            // Count characters at this position
            std::map<char, int> counts;
            int nonGapCount = 0;
            
            for (const auto& seq : profile) {
                if (i < seq.alignedData.length()) {
                    char c = seq.alignedData[i];
                    if (c != '-') {
                        counts[c]++;
                        nonGapCount++;
                    }
                }
            }
            
            if (nonGapCount == 0) continue;
            
            // Check if all characters are identical
            if (counts.size() == 1) {
                conservationLine[i] = '*';  // Fully conserved
            } else {
                // Check for conservation groups
                bool strongGroups = true;
                char firstChar = '\0';
                
                for (const auto& [c, count] : counts) {
                    if (firstChar == '\0') {
                        firstChar = c;
                    } else {
                        // Check if in same conservation group
                        if (!isInSameGroup(firstChar, c)) {
                            strongGroups = false;
                            break;
                        }
                    }
                }
                
                if (strongGroups) {
                    conservationLine[i] = ':';  // Strongly similar properties
                } else if (counts.size() <= 3) {
                    conservationLine[i] = '.';  // Weakly similar properties
                }
            }
        }
        
        return conservationLine;
    }

private:
    // Traverse the tree in post-order and perform alignments
    static void traverseAndAlign(Node* node, const std::vector<Sequence>& sequences, 
                                std::unordered_map<Node*, Profile>& nodeProfiles) {
        if (!node) return;
        
        if (node->isLeaf()) {
            // Leaf node - create a profile with a single sequence
            AlignedSequence alignedSeq(sequences[node->id]);
            nodeProfiles[node] = {alignedSeq};
        } else {
            // Internal node - align the profiles of left and right children
            traverseAndAlign(node->left, sequences, nodeProfiles);
            traverseAndAlign(node->right, sequences, nodeProfiles);
            
            // Merge the profiles from left and right
            const Profile& leftProfile = nodeProfiles[node->left];
            const Profile& rightProfile = nodeProfiles[node->right];
            
            nodeProfiles[node] = alignProfiles(leftProfile, rightProfile);
        }
    }
    
    // Needleman-Wunsch with traceback for aligning two sequences
    static std::pair<std::string, std::string> needlemanWunschAlign(
        const std::string& s1, const std::string& s2, 
        double match = 1.0, double mismatch = -1.0, double gap = -2.0) {
        
        size_t n = s1.length();
        size_t m = s2.length();
        
        // Create scoring matrix and traceback matrix
        std::vector<std::vector<double>> H(n + 1, std::vector<double>(m + 1, 0.0));
        std::vector<std::vector<char>> trace(n + 1, std::vector<char>(m + 1, 0));
        
        // Initialize first row and column with gap penalties
        for (size_t i = 0; i <= n; i++) {
            H[i][0] = i * gap;
            trace[i][0] = 'U'; // Up
        }
        
        for (size_t j = 0; j <= m; j++) {
            H[0][j] = j * gap;
            trace[0][j] = 'L'; // Left
        }
        
        // Fill the matrices
        for (size_t i = 1; i <= n; i++) {
            for (size_t j = 1; j <= m; j++) {
                double diagScore = H[i-1][j-1] + (s1[i-1] == s2[j-1] ? match : mismatch);
                double upScore = H[i-1][j] + gap;
                double leftScore = H[i][j-1] + gap;
                
                // Determine the best move
                if (diagScore >= upScore && diagScore >= leftScore) {
                    H[i][j] = diagScore;
                    trace[i][j] = 'D'; // Diagonal
                } else if (upScore >= leftScore) {
                    H[i][j] = upScore;
                    trace[i][j] = 'U'; // Up
                } else {
                    H[i][j] = leftScore;
                    trace[i][j] = 'L'; // Left
                }
            }
        }
        
        // Traceback to generate aligned sequences
        std::string aligned1;
        std::string aligned2;
        
        size_t i = n;
        size_t j = m;
        
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && trace[i][j] == 'D') {
                // Diagonal move
                aligned1 = s1[i-1] + aligned1;
                aligned2 = s2[j-1] + aligned2;
                i--; j--;
            } else if (i > 0 && (j == 0 || trace[i][j] == 'U')) {
                // Up move (gap in s2)
                aligned1 = s1[i-1] + aligned1;
                aligned2 = '-' + aligned2;
                i--;
            } else {
                // Left move (gap in s1)
                aligned1 = '-' + aligned1;
                aligned2 = s2[j-1] + aligned2;
                j--;
            }
        }
        
        return {aligned1, aligned2};
    }
    
    // Align two profiles
    static Profile alignProfiles(const Profile& profile1, const Profile& profile2) {
        if (profile1.empty()) return profile2;
        if (profile2.empty()) return profile1;
        
        size_t len1 = profile1[0].alignedData.length();
        size_t len2 = profile2[0].alignedData.length();
        
        // Create position-specific scoring matrices
        std::vector<std::map<char, double>> psm1 = createPSM(profile1);
        std::vector<std::map<char, double>> psm2 = createPSM(profile2);
        
        // Dynamic programming matrices
        std::vector<std::vector<double>> score(len1 + 1, std::vector<double>(len2 + 1, 0.0));
        std::vector<std::vector<char>> trace(len1 + 1, std::vector<char>(len2 + 1, 0));
        
        // Initialize the matrices
        for (size_t i = 0; i <= len1; i++) {
            score[i][0] = -2.0 * i;  // Gap penalty
            trace[i][0] = 'U';        // Up direction
        }
        
        for (size_t j = 0; j <= len2; j++) {
            score[0][j] = -2.0 * j;  // Gap penalty
            trace[0][j] = 'L';        // Left direction
        }
        
        // Fill the matrices
        for (size_t i = 1; i <= len1; i++) {
            for (size_t j = 1; j <= len2; j++) {
                // Calculate match score between profile columns
                double matchScore = calculateColumnScore(psm1[i-1], psm2[j-1]);
                
                double diagScore = score[i-1][j-1] + matchScore;
                double upScore = score[i-1][j] - 2.0;  // Gap penalty
                double leftScore = score[i][j-1] - 2.0; // Gap penalty
                
                // Determine the best move
                if (diagScore >= upScore && diagScore >= leftScore) {
                    score[i][j] = diagScore;
                    trace[i][j] = 'D';
                } else if (upScore >= leftScore) {
                    score[i][j] = upScore;
                    trace[i][j] = 'U';
                } else {
                    score[i][j] = leftScore;
                    trace[i][j] = 'L';
                }
            }
        }
        
        // Traceback to identify positions where gaps need to be inserted
        std::vector<int> insertGapsIntoProfile1;
        std::vector<int> insertGapsIntoProfile2;
        
        size_t i = len1;
        size_t j = len2;
        
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && trace[i][j] == 'D') {
                // Diagonal move - align columns
                i--; j--;
            } else if (i > 0 && (j == 0 || trace[i][j] == 'U')) {
                // Up move - insert gap in profile2 at column j
                insertGapsIntoProfile2.push_back(j);
                i--;
            } else {
                // Left move - insert gap in profile1 at column i
                insertGapsIntoProfile1.push_back(i);
                j--;
            }
        }
        
        // Create new merged profile
        Profile mergedProfile;
        
        // Add sequences from profile1 with gaps inserted
        for (const auto& seq : profile1) {
            std::string aligned = seq.alignedData;
            
            // Insert gaps at the specified positions (in reverse order)
            for (int pos : insertGapsIntoProfile1) {
                if (pos <= static_cast<int>(aligned.length())) {
                    aligned.insert(pos, 1, '-');
                }
            }
            
            mergedProfile.push_back(AlignedSequence(seq.id, seq.originalData, aligned));
        }
        
        // Add sequences from profile2 with gaps inserted
        for (const auto& seq : profile2) {
            std::string aligned = seq.alignedData;
            
            // Insert gaps at the specified positions (in reverse order)
            for (int pos : insertGapsIntoProfile2) {
                if (pos <= static_cast<int>(aligned.length())) {
                    aligned.insert(pos, 1, '-');
                }
            }
            
            mergedProfile.push_back(AlignedSequence(seq.id, seq.originalData, aligned));
        }
        
        return mergedProfile;
    }
    
    // Create position-specific scoring matrix for a profile
    static std::vector<std::map<char, double>> createPSM(const Profile& profile) {
        if (profile.empty()) return {};
        
        size_t length = profile[0].alignedData.length();
        std::vector<std::map<char, double>> psm(length);
        
        for (size_t pos = 0; pos < length; pos++) {
            // Count characters at this position
            std::map<char, int> counts;
            int nonGapCount = 0;
            
            for (const auto& seq : profile) {
                if (pos < seq.alignedData.length()) {
                    char c = seq.alignedData[pos];
                    if (c != '-') {
                        counts[c]++;
                        nonGapCount++;
                    }
                }
            }
            
            // Convert counts to frequencies
            if (nonGapCount > 0) {
                for (const auto& [c, count] : counts) {
                    psm[pos][c] = static_cast<double>(count) / nonGapCount;
                }
            }
        }
        
        return psm;
    }
    
    // Calculate score between two profile columns
    static double calculateColumnScore(const std::map<char, double>& col1, const std::map<char, double>& col2) {
        double score = 0.0;
        
        // If both columns are all gaps, return 0
        if (col1.empty() || col2.empty()) {
            return 0.0;
        }
        
        // Score matrix: 1.0 for match, -1.0 for mismatch
        for (const auto& [c1, freq1] : col1) {
            for (const auto& [c2, freq2] : col2) {
                double pairScore = (c1 == c2) ? 1.0 : -1.0;
                score += freq1 * freq2 * pairScore;
            }
        }
        
        return score;
    }
    
    // Check if amino acids are in the same conservation group
    static bool isInSameGroup(char a, char b) {
        // Define conservation groups for amino acids
        static const std::vector<std::string> conservationGroups = {
            "PAGST", // Small/neutral
            "ILVM",  // Hydrophobic
            "FYW",   // Aromatic
            "DE",    // Acidic
            "KRH",   // Basic
            "QN",    // Amide
            "C"      // Cysteine
        };
        
        // Convert to uppercase for consistency
        a = std::toupper(a);
        b = std::toupper(b);
        
        for (const auto& group : conservationGroups) {
            if (group.find(a) != std::string::npos && group.find(b) != std::string::npos) {
                return true;
            }
        }
        
        return false;
    }
};

// Function to print CLUSTAL format alignment
void printClustalFormat(std::ofstream& outFile, const ProgressiveAlignment::Profile& profile) {
    outFile << "CLUSTAL format alignment by MPI-MSA\n\n\n";
    
    // Calculate maximum ID length for formatting
    size_t maxIdLength = 0;
    for (size_t i = 0; i < profile.size(); i++) {
        std::string seqId = "Seq" + std::to_string(i+1);
        maxIdLength = std::max(maxIdLength, seqId.length());
    }
    
    // Print each sequence
    for (size_t i = 0; i < profile.size(); i++) {
        std::string seqId = "Seq" + std::to_string(i+1);
        outFile << std::left << std::setw(maxIdLength + 4) << seqId;
        outFile << profile[i].alignedData << std::endl;
    }
    
    // Print conservation line
    outFile << std::string(maxIdLength + 4, ' ');
    outFile << ProgressiveAlignment::generateConservationLine(profile) << std::endl;
}

// ====================== MPI Functions ======================
void computeDistanceMatrix(const std::vector<Sequence>& sequences, int rank, int numProcs, 
                          double& totalComputationTime, std::ofstream& outFile) {
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
    
    // Only root process writes to file
    if (rank == 0) {
        outFile << "Total pairs to compute: " << totalPairs << std::endl;
        outFile << "Computing pairwise alignments..." << std::endl;
    }
    
    // Synchronize all processes before timing starts
    MPI_Barrier(MPI_COMM_WORLD);
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
    
        MPI_Barrier(MPI_COMM_WORLD);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    
    // Store computation time for return value
    totalComputationTime = elapsedTime.count();
    
    // Gather computation times from all processes
    std::vector<double> procTimes(numProcs);
    MPI_Gather(&totalComputationTime, 1, MPI_DOUBLE, procTimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Gather pair counts from all processes
    std::vector<int> pairCounts(numProcs);
    int localPairCount = endPair - startPair;
    MPI_Gather(&localPairCount, 1, MPI_INT, pairCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Only rank 0 writes timing information
    if (rank == 0) {
        // Write timing for all processes
        for (int i = 0; i < numProcs; i++) {
            outFile << "Process " << i << " computed " << pairCounts[i] 
                    << " pairs in " << procTimes[i] << " seconds" << std::endl;
        }
        
        // Calculate total and average time
        double totalTime = 0.0;
        for (double time : procTimes) {
            totalTime += time;
        }
        outFile << "Average computation time: " << (totalTime / numProcs) << " seconds" << std::endl;
        outFile << "Total pairs computed: " << totalPairs << std::endl;
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
    
    // Process 0 writes the final distance matrix and results
    if (rank == 0) {
        outFile << "\nFinal Distance Matrix:" << std::endl;
        distMatrix.print(outFile);
        
        // Build and display the guide tree
        outFile << "\nConstructing guide tree using UPGMA..." << std::endl;
        GuideTree tree;
        tree.buildUPGMA(distMatrix);
        outFile << "Guide tree (Newick format): " << tree.getNewickFormat() << std::endl;
        
        // Perform progressive alignment
        outFile << "\nPerforming progressive alignment..." << std::endl;
        auto alignStart = std::chrono::high_resolution_clock::now();
        ProgressiveAlignment::Profile alignedProfile = ProgressiveAlignment::alignProgressive(sequences, tree);
        auto alignEnd = std::chrono::high_resolution_clock::now();
        double alignTime = std::chrono::duration<double>(alignEnd - alignStart).count();
        outFile << "Progressive alignment completed in " << alignTime << " seconds" << std::endl;
        
        // Print original sequences
        outFile << "\nInput Sequences:" << std::endl;
        for (size_t i = 0; i < sequences.size(); i++) {
            outFile << ">Seq" << (i+1) << " " << sequences[i].getId() << std::endl;
            // Print sequence data in lines of 60 characters (standard FASTA format)
            const std::string& seqData = sequences[i].getData();
            for (size_t j = 0; j < seqData.length(); j += 60) {
                outFile << seqData.substr(j, 60) << std::endl;
            }
        }
        
        // Print aligned sequences in CLUSTAL format
        outFile << "\n";
        printClustalFormat(outFile, alignedProfile);
        
        // Print user information with current date
        outFile << "\nAnalysis completed by: Mustafaiqbal2" << std::endl;
        outFile << "Date: 2025-03-01 10:39:02 UTC" << std::endl;
    }
}

// ====================== Main Function ======================
int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    // Output file only for rank 0
    std::ofstream outFile;
    std::string outputFilename = "msa_results.txt";
    
    if (rank == 0) {
        outFile.open(outputFilename);
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open output file: " << outputFilename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    // Synchronize to ensure file is open before proceeding
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Check command line arguments
    if (argc < 2) {
        if (rank == 0) {
            outFile << "Usage: " << argv[0] << " <fasta_file>" << std::endl;
            outFile.close();
        }
        MPI_Finalize();
        return 1;
    }
    
    std::string filename = argv[1];
    std::vector<Sequence> sequences;
    double readTime = 0.0;
    
    // Only rank 0 reads the file and broadcasts to all processes
    if (rank == 0) {
        try {
            outFile << "Reading sequences from " << filename << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            sequences = Sequence::readFasta(filename);
            auto end = std::chrono::high_resolution_clock::now();
            readTime = std::chrono::duration<double>(end - start).count();
            
            outFile << "Read " << sequences.size() << " sequences in " 
                    << readTime << " seconds" << std::endl;
        } catch (const std::exception& e) {
            outFile << "Error: " << e.what() << std::endl;
            outFile.close();
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
    double computationTime = 0.0;
    computeDistanceMatrix(sequences, rank, numProcs, computationTime, outFile);
    
    // Close the output file
    if (rank == 0) {
        outFile.close();
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}