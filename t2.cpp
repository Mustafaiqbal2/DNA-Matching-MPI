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
#include <set>
#include <array>

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
        
        // Use vector of clusters instead of just nodes
        std::vector<std::set<size_t>> clusters;
        std::vector<Node*> nodes;
        
        // Initialize with each sequence as its own cluster
        for (size_t i = 0; i < n; i++) {
            nodes.push_back(new Node(i));
            clusters.push_back({i});
        }
        
        // Copy distance matrix for manipulation
        std::vector<std::vector<double>> distances(n, std::vector<double>(n));
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                distances[i][j] = distMatrix.get(i, j);
            }
        }
        
        // Iteratively merge closest clusters
        while (clusters.size() > 1) {
            // Find minimum distance between clusters
            double minDist = std::numeric_limits<double>::max();
            size_t minI = 0, minJ = 0;
            
            for (size_t i = 0; i < clusters.size(); i++) {
                for (size_t j = i + 1; j < clusters.size(); j++) {
                    double avgDist = 0.0;
                    int count = 0;
                    
                    // Calculate average distance between all members
                    for (size_t a : clusters[i]) {
                        for (size_t b : clusters[j]) {
                            avgDist += distMatrix.get(a, b);
                            count++;
                        }
                    }
                    
                    avgDist /= count;
                    if (avgDist < minDist) {
                        minDist = avgDist;
                        minI = i;
                        minJ = j;
                    }
                }
            }
            
            // Calculate branch heights
            double height = minDist / 2.0;
            
            // Create new node
            Node* newNode = new Node(nodes[minI], nodes[minJ], height);
            
            // Merge clusters
            std::set<size_t> newCluster;
            newCluster.insert(clusters[minI].begin(), clusters[minI].end());
            newCluster.insert(clusters[minJ].begin(), clusters[minJ].end());
            
            // Remove old clusters and add new one
            if (minJ > minI) {
                clusters.erase(clusters.begin() + minJ);
                clusters.erase(clusters.begin() + minI);
                nodes.erase(nodes.begin() + minJ);
                nodes.erase(nodes.begin() + minI);
            } else {
                clusters.erase(clusters.begin() + minI);
                clusters.erase(clusters.begin() + minJ);
                nodes.erase(nodes.begin() + minI);
                nodes.erase(nodes.begin() + minJ);
            }
            
            clusters.push_back(newCluster);
            nodes.push_back(newNode);
        }
        
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
            // Include node ID and branch length (with precision)
            std::stringstream ss;
            ss << std::fixed << std::setprecision(5);
            ss << (node->id + 1) << "_Seq" << (node->id + 1) << ":" << node->height;
            return ss.str();
        } else {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(5);
            ss << "(" << newickFormat(node->left) << "," 
            << newickFormat(node->right) << "):" << node->height;
            return ss.str();
        }
    }
};

// ====================== Progressive Alignment Class ======================
class ProgressiveAlignment {
public:
    // Add a default constructor to the AlignedSequence struct
    struct AlignedSequence {
        std::string id;
        std::string originalData;
        std::string alignedData;  // With gaps inserted
        
        // Add this default constructor
        AlignedSequence() : id(""), originalData(""), alignedData("") {}
        
        AlignedSequence(const Sequence& seq) : 
            id(seq.getId()), originalData(seq.getData()), alignedData(seq.getData()) {}
        
        AlignedSequence(const std::string& id, const std::string& original, const std::string& aligned) :
            id(id), originalData(original), alignedData(aligned) {}
        
        size_t length() const { return alignedData.length(); }
    };
    
    using Profile = std::vector<AlignedSequence>;
    
    // Perform the progressive alignment based on the guide tree
    static ProgressiveAlignment::Profile alignProgressive(const std::vector<Sequence>& sequences, const GuideTree& tree) {
        Node* root = tree.getRoot();
        if (!root) {
            return {}; // Empty profile if no tree
        }
        
        // Map to store intermediate alignments at each node
        std::unordered_map<Node*, Profile> nodeProfiles;
        
        // Perform post-order traversal to build alignments from leaves to root
        traverseAndAlign(root, sequences, nodeProfiles);
        
        // Get the final profile at the root
        Profile rootProfile = nodeProfiles[root];
        
        
        
        return rootProfile;
    }

    
    // Generate conservation symbols for CLUSTAL format
    static std::string generateConservationLine(const Profile& profile) {
        if (profile.empty()) return "";
        
        size_t length = profile[0].alignedData.length();
        std::string conservationLine(length, ' ');
        
        for (size_t i = 0; i < length; i++) {
            // Count occurrences of each amino acid at this position
            std::map<char, int> counts;
            int totalCount = 0;
            int nonGapCount = 0;
            
            for (const auto& seq : profile) {
                if (i < seq.alignedData.length()) {
                    char c = seq.alignedData[i];
                    if (c != '-') {
                        counts[c]++;
                        nonGapCount++;
                    }
                    totalCount++;
                }
            }
            
            // Skip if column is all gaps
            if (nonGapCount == 0) continue;
            
            // Fully conserved if all non-gap characters are identical
            if (counts.size() == 1) {
                conservationLine[i] = '*';
                continue;
            }
            
            // Check for strong similarity (all in same group)
            bool allStrongGroups = true;
            char firstChar = '\0';
            for (const auto& [c, count] : counts) {
                if (firstChar == '\0') {
                    firstChar = c;
                } else if (!isInSameGroup(c, firstChar)) {
                    allStrongGroups = false;
                    break;
                }
            }
            
            if (allStrongGroups) {
                conservationLine[i] = ':';
                continue;
            }
            
            // Check for weak similarity (most are similar)
            int similarCount = 0;
            for (const auto& [c1, count1] : counts) {
                for (const auto& [c2, count2] : counts) {
                    if (c1 != c2 && isInSameGroup(c1, c2)) {
                        similarCount += count1 * count2;
                    }
                }
            }
            
            if (similarCount > 0 || counts.size() <= 3) {
                conservationLine[i] = '.';
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
    static double calculateEnhancedScore(const std::map<char, double>& col1, 
                                    const std::map<char, double>& col2,
                                    double match, double mismatch) {
        double score = 0.0;
        
        // If both columns are all gaps, return 0
        if (col1.empty() || col2.empty()) {
            return 0.0;
        }
        
        // Use BLOSUM-like scoring for amino acids
        for (const auto& [c1, freq1] : col1) {
            for (const auto& [c2, freq2] : col2) {
                double pairScore;
                
                if (c1 == c2) {
                    // Identical residues
                    pairScore = match;
                } else if (isInSameGroup(c1, c2)) {
                    // Similar residues (in same conservation group)
                    pairScore = 0.5;  // Half score for similar amino acids
                } else {
                    // Different residues
                    pairScore = mismatch;
                }
                
                score += freq1 * freq2 * pairScore;
            }
        }
        
        return score;
    }

        // Add this function to predict secondary structure propensities
    static void predictSecondaryStructure(const Profile& profile, std::vector<double>& helixPropensity, 
                                        std::vector<double>& sheetPropensity, std::vector<double>& loopPropensity) {
        if (profile.empty()) return;
        
        size_t length = profile[0].alignedData.length();
        helixPropensity.resize(length, 0.0);
        sheetPropensity.resize(length, 0.0);
        loopPropensity.resize(length, 0.0);
        
        // Amino acid propensities for different secondary structures
        // Values based on statistical analysis of protein structures
        // Higher values = higher propensity for that structure
        std::map<char, std::array<double, 3>> propensities = {
        {'A', std::array<double, 3>{1.45, 0.97, 0.66}}, {'R', std::array<double, 3>{1.21, 0.84, 0.95}},
        {'N', std::array<double, 3>{0.65, 0.37, 1.56}}, {'D', std::array<double, 3>{0.98, 0.53, 1.46}},
        {'C', std::array<double, 3>{0.77, 1.40, 0.98}}, {'Q', std::array<double, 3>{1.27, 0.84, 0.96}},
        {'E', std::array<double, 3>{1.44, 0.75, 0.84}}, {'G', std::array<double, 3>{0.53, 0.58, 1.72}},
        {'H', std::array<double, 3>{1.05, 0.87, 1.03}}, {'I', std::array<double, 3>{1.00, 1.60, 0.57}},
        {'L', std::array<double, 3>{1.34, 1.22, 0.57}}, {'K', std::array<double, 3>{1.23, 0.71, 1.01}},
        {'M', std::array<double, 3>{1.20, 1.11, 0.80}}, {'F', std::array<double, 3>{1.12, 1.33, 0.59}},
        {'P', std::array<double, 3>{0.57, 0.55, 1.52}}, {'S', std::array<double, 3>{0.79, 0.96, 1.18}},
        {'T', std::array<double, 3>{0.82, 1.17, 1.08}}, {'W', std::array<double, 3>{1.14, 1.19, 0.75}},
        {'Y', std::array<double, 3>{0.61, 1.42, 1.05}}, {'V', std::array<double, 3>{0.91, 1.49, 0.47}}
    };

        
        // Calculate structure propensities for each position
        for (size_t pos = 0; pos < length; pos++) {
            int nonGapCount = 0;
            double helixTotal = 0.0, sheetTotal = 0.0, loopTotal = 0.0;
            
            for (const auto& seq : profile) {
                if (pos < seq.alignedData.length()) {
                    char c = seq.alignedData[pos];
                    if (c != '-' && propensities.find(c) != propensities.end()) {
                        helixTotal += propensities[c][0];
                        sheetTotal += propensities[c][1];
                        loopTotal += propensities[c][2];
                        nonGapCount++;
                    }
                }
            }
            
            if (nonGapCount > 0) {
                helixPropensity[pos] = helixTotal / nonGapCount;
                sheetPropensity[pos] = sheetTotal / nonGapCount;
                loopPropensity[pos] = loopTotal / nonGapCount;
            }
        }
        
        // Apply a sliding window to smooth the propensities
        const int windowSize = 4;
        std::vector<double> smoothedHelix(length), smoothedSheet(length), smoothedLoop(length);
        
        for (size_t pos = 0; pos < length; pos++) {
            double helixSum = 0.0, sheetSum = 0.0, loopSum = 0.0;
            int count = 0;
            
            for (int offset = -windowSize; offset <= windowSize; offset++) {
                int checkPos = pos + offset;
                if (checkPos >= 0 && checkPos < static_cast<int>(length)) {
                    helixSum += helixPropensity[checkPos];
                    sheetSum += sheetPropensity[checkPos];
                    loopSum += loopPropensity[checkPos];
                    count++;
                }
            }
            
            if (count > 0) {
                smoothedHelix[pos] = helixSum / count;
                smoothedSheet[pos] = sheetSum / count;
                smoothedLoop[pos] = loopSum / count;
            }
        }
        
        // Update with smoothed values
        helixPropensity = smoothedHelix;
        sheetPropensity = smoothedSheet;
        loopPropensity = smoothedLoop;
    }

    // Modify the alignProfiles function to use secondary structure awareness for gap penalties
    static double getStructureAwareGapPenalty(double baseGapOpen, double baseGapExtend, bool isOpening,
                                            const std::vector<double>& helixProp,
                                            const std::vector<double>& sheetProp,
                                            const std::vector<double>& loopProp,
                                            size_t position) {
        if (position >= helixProp.size()) {
            return isOpening ? baseGapOpen : baseGapExtend;
        }
        
        // Determine which structure has highest propensity
        double maxProp = std::max({helixProp[position], sheetProp[position], loopProp[position]});
        
        if (loopProp[position] >= maxProp * 0.9) {
            // In loop regions, make gaps more likely
            return isOpening ? baseGapOpen * 0.7 : baseGapExtend * 0.7;
        } else if (helixProp[position] >= maxProp * 0.9 || sheetProp[position] >= maxProp * 0.9) {
            // In structured regions, discourage gaps
            return isOpening ? baseGapOpen * 1.5 : baseGapExtend * 1.3;
        }
        
        return isOpening ? baseGapOpen : baseGapExtend;
    }

        // Add BLOSUM62 scoring matrix
    static std::map<std::pair<char, char>, int> getBLOSUM62() {
        static std::map<std::pair<char, char>, int> blosum62 = {
            {{'A','A'}, 4}, {{'A','R'}, -1}, {{'A','N'}, -2}, {{'A','D'}, -2}, {{'A','C'}, 0},
            {{'A','Q'}, -1}, {{'A','E'}, -1}, {{'A','G'}, 0}, {{'A','H'}, -2}, {{'A','I'}, -1},
            {{'A','L'}, -1}, {{'A','K'}, -1}, {{'A','M'}, -1}, {{'A','F'}, -2}, {{'A','P'}, -1},
            {{'A','S'}, 1}, {{'A','T'}, 0}, {{'A','W'}, -3}, {{'A','Y'}, -2}, {{'A','V'}, 0},
            {{'R','R'}, 5}, {{'R','N'}, 0}, {{'R','D'}, -2}, {{'R','C'}, -3}, {{'R','Q'}, 1},
            {{'R','E'}, 0}, {{'R','G'}, -2}, {{'R','H'}, 0}, {{'R','I'}, -3}, {{'R','L'}, -2},
            {{'R','K'}, 2}, {{'R','M'}, -1}, {{'R','F'}, -3}, {{'R','P'}, -2}, {{'R','S'}, -1},
            {{'R','T'}, -1}, {{'R','W'}, -3}, {{'R','Y'}, -2}, {{'R','V'}, -3}, {{'N','N'}, 6},
            {{'N','D'}, 1}, {{'N','C'}, -3}, {{'N','Q'}, 0}, {{'N','E'}, 0}, {{'N','G'}, 0},
            {{'N','H'}, 1}, {{'N','I'}, -3}, {{'N','L'}, -3}, {{'N','K'}, 0}, {{'N','M'}, -2},
            {{'N','F'}, -3}, {{'N','P'}, -2}, {{'N','S'}, 1}, {{'N','T'}, 0}, {{'N','W'}, -4},
            {{'N','Y'}, -2}, {{'N','V'}, -3}, {{'D','D'}, 6}, {{'D','C'}, -3}, {{'D','Q'}, 0},
            {{'D','E'}, 2}, {{'D','G'}, -1}, {{'D','H'}, -1}, {{'D','I'}, -3}, {{'D','L'}, -4},
            {{'D','K'}, -1}, {{'D','M'}, -3}, {{'D','F'}, -3}, {{'D','P'}, -1}, {{'D','S'}, 0},
            {{'D','T'}, -1}, {{'D','W'}, -4}, {{'D','Y'}, -3}, {{'D','V'}, -3}, {{'C','C'}, 9},
            {{'C','Q'}, -3}, {{'C','E'}, -4}, {{'C','G'}, -3}, {{'C','H'}, -3}, {{'C','I'}, -1},
            {{'C','L'}, -1}, {{'C','K'}, -3}, {{'C','M'}, -1}, {{'C','F'}, -2}, {{'C','P'}, -3},
            {{'C','S'}, -1}, {{'C','T'}, -1}, {{'C','W'}, -2}, {{'C','Y'}, -2}, {{'C','V'}, -1},
            {{'Q','Q'}, 5}, {{'Q','E'}, 2}, {{'Q','G'}, -2}, {{'Q','H'}, 0}, {{'Q','I'}, -3},
            {{'Q','L'}, -2}, {{'Q','K'}, 1}, {{'Q','M'}, 0}, {{'Q','F'}, -3}, {{'Q','P'}, -1},
            {{'Q','S'}, 0}, {{'Q','T'}, -1}, {{'Q','W'}, -2}, {{'Q','Y'}, -1}, {{'Q','V'}, -2},
            {{'E','E'}, 5}, {{'E','G'}, -2}, {{'E','H'}, 0}, {{'E','I'}, -3}, {{'E','L'}, -3},
            {{'E','K'}, 1}, {{'E','M'}, -2}, {{'E','F'}, -3}, {{'E','P'}, -1}, {{'E','S'}, 0},
            {{'E','T'}, -1}, {{'E','W'}, -3}, {{'E','Y'}, -2}, {{'E','V'}, -2}, {{'G','G'}, 6},
            {{'G','H'}, -2}, {{'G','I'}, -4}, {{'G','L'}, -4}, {{'G','K'}, -2}, {{'G','M'}, -3},
            {{'G','F'}, -3}, {{'G','P'}, -2}, {{'G','S'}, 0}, {{'G','T'}, -2}, {{'G','W'}, -2},
            {{'G','Y'}, -3}, {{'G','V'}, -3}, {{'H','H'}, 8}, {{'H','I'}, -3}, {{'H','L'}, -3},
            {{'H','K'}, -1}, {{'H','M'}, -2}, {{'H','F'}, -1}, {{'H','P'}, -2}, {{'H','S'}, -1},
            {{'H','T'}, -2}, {{'H','W'}, -2}, {{'H','Y'}, 2}, {{'H','V'}, -3}, {{'I','I'}, 4},
            {{'I','L'}, 2}, {{'I','K'}, -3}, {{'I','M'}, 1}, {{'I','F'}, 0}, {{'I','P'}, -3},
            {{'I','S'}, -2}, {{'I','T'}, -1}, {{'I','W'}, -3}, {{'I','Y'}, -1}, {{'I','V'}, 3},
            {{'L','L'}, 4}, {{'L','K'}, -2}, {{'L','M'}, 2}, {{'L','F'}, 0}, {{'L','P'}, -3},
            {{'L','S'}, -2}, {{'L','T'}, -1}, {{'L','W'}, -2}, {{'L','Y'}, -1}, {{'L','V'}, 1},
            {{'K','K'}, 5}, {{'K','M'}, -1}, {{'K','F'}, -3}, {{'K','P'}, -1}, {{'K','S'}, 0},
            {{'K','T'}, -1}, {{'K','W'}, -3}, {{'K','Y'}, -2}, {{'K','V'}, -2}, {{'M','M'}, 5},
            {{'M','F'}, 0}, {{'M','P'}, -2}, {{'M','S'}, -1}, {{'M','T'}, -1}, {{'M','W'}, -1},
            {{'M','Y'}, -1}, {{'M','V'}, 1}, {{'F','F'}, 6}, {{'F','P'}, -4}, {{'F','S'}, -2},
            {{'F','T'}, -2}, {{'F','W'}, 1}, {{'F','Y'}, 3}, {{'F','V'}, -1}, {{'P','P'}, 7},
            {{'P','S'}, -1}, {{'P','T'}, -1}, {{'P','W'}, -4}, {{'P','Y'}, -3}, {{'P','V'}, -2},
            {{'S','S'}, 4}, {{'S','T'}, 1}, {{'S','W'}, -3}, {{'S','Y'}, -2}, {{'S','V'}, -2},
            {{'T','T'}, 5}, {{'T','W'}, -2}, {{'T','Y'}, -2}, {{'T','V'}, 0}, {{'W','W'}, 11},
            {{'W','Y'}, 2}, {{'W','V'}, -3}, {{'Y','Y'}, 7}, {{'Y','V'}, -1}, {{'V','V'}, 4}
        };
        
        // Make sure all pairs of amino acids are covered (symmetric matrix)
        std::map<std::pair<char, char>, int> completeMatrix;
        for (const auto& entry : blosum62) {
            char a = entry.first.first;
            char b = entry.first.second;
            int score = entry.second;
            
            completeMatrix[{a, b}] = score;
            if (a != b) {
                completeMatrix[{b, a}] = score; // Add symmetric entry
            }
        }
        
        return completeMatrix;
    }

    // Calculate score between positions using BLOSUM62
    static double calculateBLOSUM62Score(const std::map<char, double>& col1, const std::map<char, double>& col2) {
        static const auto blosum62 = getBLOSUM62();
        
        if (col1.empty() || col2.empty()) {
            return 0.0; // All gaps
        }
        
        double score = 0.0;
        
        // Calculate the weighted score based on amino acid frequencies
        for (const auto& p1 : col1) {
            char aa1 = p1.first;
            double freq1 = p1.second;
            
            for (const auto& p2 : col2) {
                char aa2 = p2.first;
                double freq2 = p2.second;
                
                // Find the BLOSUM62 score for this pair
                auto key = std::make_pair(aa1, aa2);
                auto it = blosum62.find(key);
                
                if (it != blosum62.end()) {
                    // Use BLOSUM62 score
                    score += freq1 * freq2 * it->second;
                } else {
                    // Fallback if not found in BLOSUM62 (like for special characters)
                    score += freq1 * freq2 * (aa1 == aa2 ? 1.0 : -1.0);
                }
            }
        }
        
        return score;
    }
    
    // Align two profiles
    // Implement a proper three-matrix dynamic programming algorithm for affine gap penalties
    static Profile alignProfiles(const Profile& profile1, const Profile& profile2) {
        if (profile1.empty()) return profile2;
        if (profile2.empty()) return profile1;
        
        size_t len1 = profile1[0].alignedData.length();
        size_t len2 = profile2[0].alignedData.length();
        
        // Create position-specific scoring matrices with improved weighting
        std::vector<std::map<char, double>> psm1 = createPSM(profile1);
        std::vector<std::map<char, double>> psm2 = createPSM(profile2);
        
        // Gap penalty parameters
        double gapOpen = -10.0;    // Penalty for opening a gap
        double gapExtend = -1.0;   // Penalty for extending a gap
        
        // Three matrices for proper affine gap penalties
        std::vector<std::vector<double>> M(len1 + 1, std::vector<double>(len2 + 1, -std::numeric_limits<double>::infinity())); // Match/mismatch
        std::vector<std::vector<double>> X(len1 + 1, std::vector<double>(len2 + 1, -std::numeric_limits<double>::infinity())); // Gap in profile1
        std::vector<std::vector<double>> Y(len1 + 1, std::vector<double>(len2 + 1, -std::numeric_limits<double>::infinity())); // Gap in profile2
        
        // Traceback matrices
        std::vector<std::vector<char>> traceM(len1 + 1, std::vector<char>(len2 + 1, 0)); // Traceback for M
        std::vector<std::vector<char>> traceX(len1 + 1, std::vector<char>(len2 + 1, 0)); // Traceback for X
        std::vector<std::vector<char>> traceY(len1 + 1, std::vector<char>(len2 + 1, 0)); // Traceback for Y
        
        // Initialize matrices
        M[0][0] = 0;
        X[0][0] = -std::numeric_limits<double>::infinity();
        Y[0][0] = -std::numeric_limits<double>::infinity();
        
        // Initialize first column
        for (size_t i = 1; i <= len1; i++) {
            M[i][0] = -std::numeric_limits<double>::infinity();
            X[i][0] = gapOpen + (i-1) * gapExtend;
            Y[i][0] = -std::numeric_limits<double>::infinity();
            
            traceX[i][0] = 'X'; // Continue in X
        }
        
        // Initialize first row
        for (size_t j = 1; j <= len2; j++) {
            M[0][j] = -std::numeric_limits<double>::infinity();
            X[0][j] = -std::numeric_limits<double>::infinity();
            Y[0][j] = gapOpen + (j-1) * gapExtend;
            
            traceY[0][j] = 'Y'; // Continue in Y
        }

            // First calculate the structure propensities:
                std::vector<double> helixProp1, sheetProp1, loopProp1;
                std::vector<double> helixProp2, sheetProp2, loopProp2;
                predictSecondaryStructure(profile1, helixProp1, sheetProp1, loopProp1);
                predictSecondaryStructure(profile2, helixProp2, sheetProp2, loopProp2);

        
        // Fill the matrices
        for (size_t i = 1; i <= len1; i++) {
            for (size_t j = 1; j <= len2; j++) {


                // Calculate match score using position-specific scoring
                double matchScore = calculateBLOSUM62Score(psm1[i-1], psm2[j-1]);
                
                // M matrix - ending with a match/mismatch
                double mFromM = M[i-1][j-1] + getStructureAwareGapPenalty(gapOpen, gapExtend, false, 
                                                                    helixProp1, sheetProp1, loopProp1, i-1) + matchScore;
                double mFromX = X[i-1][j-1] + getStructureAwareGapPenalty(gapOpen, gapExtend, true,
                                                                    helixProp1, sheetProp1, loopProp1, i-1) + matchScore;   
                double mFromY = Y[i-1][j-1] + getStructureAwareGapPenalty(gapOpen, gapExtend, true,
                                                                    helixProp1, sheetProp1, loopProp1, i-1) + matchScore;
                
                if (mFromM >= mFromX && mFromM >= mFromY) {
                    M[i][j] = mFromM;
                    traceM[i][j] = 'M';
                } else if (mFromX >= mFromY) {
                    M[i][j] = mFromX;
                    traceM[i][j] = 'X';
                } else {
                    M[i][j] = mFromY;
                    traceM[i][j] = 'Y';
                }
                
                // X matrix - ending with a gap in profile2
                // Usage in the DP matrix fill:

                // Then adjust the gap penalties in matrix filling:
                double xFromM = M[i-1][j] + getStructureAwareGapPenalty(gapOpen, gapExtend, true, 
                                                                    helixProp1, sheetProp1, loopProp1, i-1);
                double xFromX = X[i-1][j] + getStructureAwareGapPenalty(gapOpen, gapExtend, false, 
                                                                    helixProp1, sheetProp1, loopProp1, i-1);

                                
                if (xFromM >= xFromX) {
                    X[i][j] = xFromM;
                    traceX[i][j] = 'M';
                } else {
                    X[i][j] = xFromX;
                    traceX[i][j] = 'X';
                }
                
                // Y matrix - ending with a gap in profile1
                double yFromM = M[i][j-1] + getStructureAwareGapPenalty(gapOpen, gapExtend, true, 
                                                        helixProp2, sheetProp2, loopProp2, j-1);
                double yFromY = Y[i][j-1] + getStructureAwareGapPenalty(gapOpen, gapExtend, false, 
                                                        helixProp2, sheetProp2, loopProp2, j-1);
                
                if (yFromM >= yFromY) {
                    Y[i][j] = yFromM;
                    traceY[i][j] = 'M';
                } else {
                    Y[i][j] = yFromY;
                    traceY[i][j] = 'Y';
                }
            }
        }
        
        // Find which matrix has the best score at the end
        double bestScore = M[len1][len2];
        char currentMatrix = 'M';
        
        if (X[len1][len2] > bestScore) {
            bestScore = X[len1][len2];
            currentMatrix = 'X';
        }
        
        if (Y[len1][len2] > bestScore) {
            bestScore = Y[len1][len2];
            currentMatrix = 'Y';
        }
        
        // Traceback to identify positions where gaps need to be inserted
        std::vector<int> insertGapsIntoProfile1;
        std::vector<int> insertGapsIntoProfile2;
        
        size_t i = len1;
        size_t j = len2;
        
        while (i > 0 || j > 0) {
            if (currentMatrix == 'M') {
                // Match state - align both positions
                char nextMatrix = traceM[i][j];
                i--;
                j--;
                currentMatrix = nextMatrix;
            } else if (currentMatrix == 'X') {
                // X state - gap in profile2
                char nextMatrix = traceX[i][j];
                insertGapsIntoProfile2.push_back(j);
                i--;
                currentMatrix = nextMatrix;
            } else { // currentMatrix == 'Y'
                // Y state - gap in profile1
                char nextMatrix = traceY[i][j];
                insertGapsIntoProfile1.push_back(i);
                j--;
                currentMatrix = nextMatrix;
            }
        }
        
        // Create new merged profile
        Profile mergedProfile;
        
        // Add sequences from profile1 with gaps inserted
        for (const auto& seq : profile1) {
            std::string aligned = seq.alignedData;
            
            // Insert gaps in reverse order to maintain correct positions
            std::sort(insertGapsIntoProfile1.begin(), insertGapsIntoProfile1.end(), std::greater<int>());
            
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
            
            // Insert gaps in reverse order to maintain correct positions
            std::sort(insertGapsIntoProfile2.begin(), insertGapsIntoProfile2.end(), std::greater<int>());
            
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
    size_t maxIdLength = 7;  // "Seq" + digit + padding
    
    // Print each sequence
    for (const auto& seq : profile) {
        // Extract sequence number from ID
        std::string seqId = "Seq";
        size_t idPos = seq.id.find_first_of("0123456789");
        if (idPos != std::string::npos) {
            seqId += seq.id.substr(idPos);
        }
        
        outFile << std::left << std::setw(maxIdLength) << seqId;
        outFile << seq.alignedData << std::endl;
    }
    
    // Print conservation line
    outFile << std::string(maxIdLength, ' ');
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