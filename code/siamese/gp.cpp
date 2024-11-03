// ContrastiveGP.hpp
#pragma once

#include <OpenXLSX.hpp>
#include <unordered_set>
#include <vector>
#include <random>
#include <memory>
#include <cmath>
#include <algorithm>
#include <thread>
#include <future>
#include <map>
#include <functional>
#include <string>
#include <iostream>

// Configuration struct
struct GPConfig {
    int n_features = 2080;
    int num_trees = 10;
    int population_size = 100;
    int generations = 100;
    int elite_size = 10;
    float crossover_prob = 0.8f;
    float mutation_prob = 0.3f;
    int tournament_size = 7;
    float distance_threshold = 1.0f;
    float margin = 1.0f;
    float fitness_alpha = 0.9f;
    float loss_alpha = 0.1f;
    float parsimony_coeff = 0.01f;
    int max_tree_depth = 6;
    int batch_size = 128;
    int num_workers = std::thread::hardware_concurrency();  // Use available CPU cores
    float dropout_prob = 0.001f;
    float bn_momentum = 0.1f;
    float bn_epsilon = 1e-5f;
};

// Data structures
struct DataPoint {
    std::vector<float> anchor;
    std::vector<float> compare;
    float label;

    DataPoint(const std::vector<float>& a, const std::vector<float>& c, float l)
        : anchor(a), compare(c), label(l) {}

    DataPoint(const std::vector<float>& features, float l)
    : anchor(features), compare(features), label(l) {}
};

class ExcelProcessor {
private:
    // Set of labels to discard
    const std::unordered_set<std::string> discardLabels = {
        "HM", "QC", "MO", "Fillet", "Frame", "Skins", 
        "Livers", "Guts", "Gonads", "Heads"
    };

    struct Instance {
        std::string label;
        std::vector<float> features;

        Instance(std::string l, std::vector<float> f) 
            : label(std::move(l)), features(std::move(f)) {}
    };

    // Helper function to check if a label should be discarded
    bool shouldDiscard(const std::string& label) {
        for (const auto& discardLabel : discardLabels) {
            if (label.find(discardLabel) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    // Helper function to generate all possible pairs from instances
    // Modified generatePairs function for ExcelProcessor class
    std::vector<DataPoint> generatePairs(const std::vector<Instance>& instances, 
                                   size_t maxPairsPerClass = 50,  // Default value set to 50
                                   size_t maxNegativePairs = 50)  { // Also limit negative pairs
        std::vector<DataPoint> pairs;
        std::random_device rd;
        std::mt19937 gen(rd());

        // Group instances by label
        std::unordered_map<std::string, std::vector<const Instance*>> labelGroups;
        for (const auto& instance : instances) {
            labelGroups[instance.label].push_back(&instance);
        }

        // Generate positive pairs (same label)
        int total_positive_pairs = 0;
        for (const auto& [label, group] : labelGroups) {
            if (group.size() < 2) continue;

            // Generate all possible pairs
            std::vector<std::pair<size_t, size_t>> possiblePairs;
            for (size_t i = 0; i < group.size(); ++i) {
                for (size_t j = i + 1; j < group.size(); ++j) {
                    possiblePairs.emplace_back(i, j);
                }
            }

            // Shuffle and limit number of pairs per class
            std::shuffle(possiblePairs.begin(), possiblePairs.end(), gen);
            size_t numPairs = std::min(possiblePairs.size(), maxPairsPerClass);

            for (size_t i = 0; i < numPairs; ++i) {
                const auto& [idx1, idx2] = possiblePairs[i];
                pairs.emplace_back(group[idx1]->features, 
                                group[idx2]->features, 
                                1.0f);
                total_positive_pairs++;
            }

            std::cout << "Generated " << numPairs << " positive pairs for class " << label << std::endl;
        }

        // Generate negative pairs (different labels)
        std::vector<const Instance*> allInstances;
        for (const auto& instance : instances) {
            allInstances.push_back(&instance);
        }

        // Generate balanced number of negative pairs
        size_t desired_negative_pairs = std::min(total_positive_pairs, static_cast<int>(maxNegativePairs));
        std::uniform_int_distribution<size_t> dist(0, allInstances.size() - 1);
        
        int negative_pairs_generated = 0;
        int max_attempts = desired_negative_pairs * 10;  // Avoid infinite loop
        int attempts = 0;

        std::cout << "Attempting to generate " << desired_negative_pairs << " negative pairs..." << std::endl;

        while (negative_pairs_generated < desired_negative_pairs && attempts < max_attempts) {
            size_t idx1 = dist(gen);
            size_t idx2 = dist(gen);
            attempts++;

            if (idx1 != idx2 && allInstances[idx1]->label != allInstances[idx2]->label) {
                pairs.emplace_back(allInstances[idx1]->features,
                                allInstances[idx2]->features,
                                0.0f);
                negative_pairs_generated++;
            }
        }

        std::cout << "Generated " << negative_pairs_generated << " negative pairs" << std::endl;
        std::cout << "Total pairs generated: " << pairs.size() << std::endl;

        // Final shuffle of all pairs
        std::shuffle(pairs.begin(), pairs.end(), gen);
        return pairs;
    }

public:
    std::vector<DataPoint> readExcel(const std::string& filename, 
                               const std::string& sheetName = "All data no QC filtering") {
        try {
            std::cout << "Opening Excel file: " << filename << std::endl;
            OpenXLSX::XLDocument doc;
            doc.open(filename);
            
            std::cout << "Getting worksheet: " << sheetName << std::endl;
            OpenXLSX::XLWorksheet wks = doc.workbook().worksheet(sheetName);
            
            // Find actual data range by scanning the first column
            uint32_t actualLastRow = 0;
            uint32_t actualLastCol = 0;

            // First find the last row with data in column 1
            for (uint32_t row = 1; row <= 1000; ++row) {  // Set a reasonable upper limit
                auto cell = wks.cell(row, 1);
                if (cell.value().type() == OpenXLSX::XLValueType::Empty) {
                    actualLastRow = row - 1;
                    break;
                }
            }

            // Then find the last column with data in row 1 (header row)
            for (uint32_t col = 1; col <= 3000; ++col) {  // Set a reasonable upper limit
                auto cell = wks.cell(1, col);
                if (cell.value().type() == OpenXLSX::XLValueType::Empty) {
                    actualLastCol = col - 1;
                    break;
                }
            }

            std::cout << "Found actual data dimensions: " << actualLastRow << " rows and " 
                    << actualLastCol << " columns" << std::endl;
            
            if (actualLastRow < 2 || actualLastCol < 2) {
                throw std::runtime_error("Excel file has insufficient data");
            }

            std::vector<Instance> instances;
            instances.reserve(actualLastRow - 1);  // Reserve space excluding header row
            
            // Process data rows
            for (uint32_t row = 2; row <= actualLastRow; ++row) {
                try {
                    auto labelCell = wks.cell(row, 1);
                    
                    // Skip if cell is empty
                    if (labelCell.value().type() == OpenXLSX::XLValueType::Empty) {
                        continue;
                    }

                    std::string label = labelCell.value().get<std::string>();
                    
                    // Skip if label is empty or should be discarded
                    if (label.empty() || shouldDiscard(label)) {
                        continue;
                    }
                    
                    // Read features
                    std::vector<float> features;
                    features.reserve(actualLastCol - 1);
                    bool validRow = true;
                    
                    for (uint32_t col = 2; col <= actualLastCol; ++col) {
                        auto cell = wks.cell(row, col);
                        auto cellType = cell.value().type();
                        
                        if (cellType == OpenXLSX::XLValueType::Empty) {
                            validRow = false;
                            break;  // Skip rows with any empty feature cells
                        }
                        
                        try {
                            float value = 0.0f;
                            if (cellType == OpenXLSX::XLValueType::Float) {
                                value = static_cast<float>(cell.value().get<double>());
                            }
                            else if (cellType == OpenXLSX::XLValueType::Integer) {
                                value = static_cast<float>(cell.value().get<int64_t>());
                            }
                            else {
                                validRow = false;
                                break;  // Skip rows with non-numeric feature cells
                            }
                            
                            features.push_back(value);
                        }
                        catch (const std::exception&) {
                            validRow = false;
                            break;
                        }
                    }
                    
                    // Only add instance if all features are valid
                    if (validRow && features.size() == actualLastCol - 1) {
                        instances.emplace_back(std::move(label), std::move(features));
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Warning: Error processing row " << row 
                            << ": " << e.what() << std::endl;
                    continue;
                }
            }
            
            std::cout << "Successfully processed " << instances.size() 
                    << " valid instances" << std::endl;
            
            // Generate pairs before closing the document
            std::cout << "Generating pairs..." << std::endl;
            auto pairs = generatePairs(instances);
            std::cout << "Generated " << pairs.size() << " pairs" << std::endl;
            
            doc.close();
            return pairs;
            
        } catch (const OpenXLSX::XLException& e) {
            throw std::runtime_error("Excel error: " + std::string(e.what()));
        } catch (const std::exception& e) {
            throw std::runtime_error("Error reading Excel file: " + std::string(e.what()));
        }
    }

    // Splits data into training and validation sets
    std::pair<std::vector<DataPoint>, std::vector<DataPoint>> 
    splitTrainVal(const std::vector<DataPoint>& data, float valRatio = 0.2) {
        size_t valSize = static_cast<size_t>(data.size() * valRatio);
        size_t trainSize = data.size() - valSize;

        std::vector<DataPoint> trainData(data.begin(), data.begin() + trainSize);
        std::vector<DataPoint> valData(data.begin() + trainSize, data.end());

        return {trainData, valData};
    }
};

// Vector operations helper functions
namespace vec_ops {
    inline std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    inline std::vector<float> subtract(const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    inline std::vector<float> multiply(const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] * b[i];
        }
        return result;
    }

    inline float mean(const std::vector<float>& v) {
        if (v.empty()) return 0.0f;
        return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    }

    inline float variance(const std::vector<float>& v, float mean) {
        if (v.empty()) return 0.0f;
        float sum = 0.0f;
        for (float x : v) {
            float diff = x - mean;
            sum += diff * diff;
        }
        return sum / v.size();
    }
}

class BatchNorm {
private:
    float momentum;
    float epsilon;
    float running_mean;
    float running_var;
    bool training;

public:
    BatchNorm(float m = 0.1f, float e = 1e-5f)
        : momentum(m), epsilon(e), running_mean(0.0f), running_var(1.0f), training(true) {
    }

    void setTraining(bool mode) { training = mode; }

    std::vector<float> operator()(const std::vector<float>& x) {
        if (x.empty()) {
            std::cout << "BatchNorm: Empty input" << std::endl;
            return x;
        }

        // For single values, just do basic standardization
        if (x.size() == 1) {
            float value = x[0];
            if (training) {
                running_mean = (1 - momentum) * running_mean + momentum * value;
                running_var = (1 - momentum) * running_var + 
                            momentum * (value - running_mean) * (value - running_mean);
            }
            
            // Use a larger epsilon for numerical stability
            float std_dev = std::sqrt(running_var + 0.1f);  // Increased epsilon
            float normalized = (value - running_mean) / std_dev;
            
            return std::vector<float>{normalized};
        }

        // For vectors, compute proper batch statistics
        float batch_mean = 0.0f;
        for (float val : x) {
            batch_mean += val;
        }
        batch_mean /= x.size();

        float batch_var = 0.0f;
        for (float val : x) {
            float diff = val - batch_mean;
            batch_var += diff * diff;
        }
        batch_var = batch_var / x.size() + epsilon;

        if (training) {
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean;
            running_var = (1 - momentum) * running_var + momentum * batch_var;
        }

        std::vector<float> result(x.size());
        float std_dev = std::sqrt(training ? batch_var : running_var + epsilon);
        float mean = training ? batch_mean : running_mean;

        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = (x[i] - mean) / std_dev;
        }

        return result;
    }
};

// GP Operations class
class GPOperations {
public:
    // Constructor
    GPOperations(float dropout_probability = 0.0f,
                float batch_norm_momentum = 0.01f,
                float batch_norm_epsilon = 0.1f)
        : dropout_prob(dropout_probability)
        , training(true)
        , current_eval_id(0)
    {
        // Initialize batch normalizations for each operation type
        initializeBatchNorms(batch_norm_momentum, batch_norm_epsilon);
    }

    // Disable copying
    GPOperations(const GPOperations&) = delete;
    GPOperations& operator=(const GPOperations&) = delete;

    // Main evaluation interface
    std::vector<float> evaluate(const std::string& op_name,
                              const std::vector<std::vector<float>>& inputs,
                              size_t node_id) {
        std::lock_guard<std::mutex> lock(mutex);

        // Check for dropout
        if (shouldDropNode(node_id)) {
            return scaleForDropout(inputs[0]);
        }

        // Validate inputs
        if (inputs.empty() || inputs[0].empty()) {
            return std::vector<float>();
        }

        // Perform operation
        std::vector<float> result = performOperation(op_name, inputs);

        // Apply batch normalization
        if (auto it = batch_norms.find(op_name); it != batch_norms.end()) {
            result = it->second->operator()(result);
        }

        return result;
    }

    // Training mode control
    void setTraining(bool mode) {
        std::lock_guard<std::mutex> lock(mutex);
        training = mode;
        for (auto& [_, bn] : batch_norms) {
            bn->setTraining(mode);
        }
    }

    // Start new evaluation cycle
    void startNewEvaluation() {
        std::lock_guard<std::mutex> lock(mutex);
        ++current_eval_id;
        dropout_mask.clear();
    }

private:
    // Member variables
    float dropout_prob;
    bool training;
    size_t current_eval_id;
    std::mutex mutex;
    std::map<std::string, std::unique_ptr<BatchNorm>> batch_norms;
    std::map<size_t, bool> dropout_mask;
    static thread_local std::mt19937 rng;

    // Initialize batch normalizations
    void initializeBatchNorms(float momentum, float epsilon) {
        std::vector<std::string> op_types = {
            "add", "sub", "mul", "div", "sin", "cos", "neg"
        };
        for (const auto& op : op_types) {
            batch_norms[op] = std::make_unique<BatchNorm>(momentum, epsilon);
        }
    }

    // Dropout check
    bool shouldDropNode(size_t node_id) {
        if (!training || dropout_prob <= 0.0f) {
            return false;
        }

        size_t mask_key = (current_eval_id << 32) | node_id;
        if (dropout_mask.find(mask_key) == dropout_mask.end()) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            dropout_mask[mask_key] = dist(rng) < dropout_prob;
        }
        return dropout_mask[mask_key];
    }

    // Scale outputs when using dropout
    std::vector<float> scaleForDropout(const std::vector<float>& x) const {
        if (dropout_prob <= 0.0f) return x;

        std::vector<float> scaled = x;
        float scale = 1.0f / (1.0f - dropout_prob);
        for (float& val : scaled) {
            val *= scale;
        }
        return scaled;
    }

    // Core operation implementation
    std::vector<float> performOperation(const std::string& op_name,
                                      const std::vector<std::vector<float>>& inputs) {
        if (op_name == "add") {
            return vec_ops::add(inputs[0], inputs[1]);
        }
        else if (op_name == "sub") {
            return vec_ops::subtract(inputs[0], inputs[1]);
        }
        else if (op_name == "mul") {
            return vec_ops::multiply(inputs[0], inputs[1]);
        }
        else if (op_name == "div") {
            return protectedDiv(inputs[0], inputs[1]);
        }
        else if (op_name == "sin") {
            return applySin(inputs[0]);
        }
        else if (op_name == "cos") {
            return applyCos(inputs[0]);
        }
        else if (op_name == "neg") {
            return applyNeg(inputs[0]);
        }
        
        // Default case: return empty vector
        return std::vector<float>();
    }

    // Protected operations
    std::vector<float> protectedDiv(const std::vector<float>& x,
                                  const std::vector<float>& y) {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::abs(y[i]) < 1e-10f ? 
                       x[i] : x[i] / (y[i] + 1e-10f);
        }
        return result;
    }

    std::vector<float> applySin(const std::vector<float>& x) {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::sin(x[i]);
        }
        return result;
    }

    std::vector<float> applyCos(const std::vector<float>& x) {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::cos(x[i]);
        }
        return result;
    }

    std::vector<float> applyNeg(const std::vector<float>& x) {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = -x[i];
        }
        return result;
    }
};

// GP Node class hierarchy
class GPNode {
public:
    virtual ~GPNode() = default;
    virtual std::vector<float> evaluate(const std::vector<float>& input) = 0;
    virtual std::unique_ptr<GPNode> clone() const = 0;
    virtual int size() const = 0;
    virtual int depth() const = 0;
    virtual void mutate(std::mt19937& gen, const GPConfig& config) = 0;
};

class FeatureNode : public GPNode {
private:
    int feature_index;
    static constexpr float SCALE_FACTOR = 0.01f;  // Scale factor to bring values to [-1, 1] range

public:
    explicit FeatureNode(int idx) : feature_index(idx) {}

    std::vector<float> evaluate(const std::vector<float>& input) override {
        if (feature_index >= input.size()) {
            std::cerr << "Warning: Feature index " << feature_index 
                      << " out of bounds for input size " << input.size() << std::endl;
            return std::vector<float>{0.0f};
        }
        
        // Scale the input value to a reasonable range
        float raw_value = input[feature_index];
        float scaled_value = std::tanh(raw_value * SCALE_FACTOR);
        
        return std::vector<float>{scaled_value};
    }

    std::unique_ptr<GPNode> clone() const override {
        return std::make_unique<FeatureNode>(feature_index);
    }

    int size() const override { return 1; }
    int depth() const override { return 1; }

    void mutate(std::mt19937& gen, const GPConfig& config) override {
        std::uniform_int_distribution<int> dist(0, config.n_features - 1);
        feature_index = dist(gen);
    }
};

// Also modify the constant node to stay in a reasonable range
class ConstantNode : public GPNode {
private:
    float value;

public:
    explicit ConstantNode(float v) : value(std::tanh(v)) {}  // Ensure value is in [-1, 1]

    std::vector<float> evaluate(const std::vector<float>& input) override {
        return std::vector<float>{value};
    }

    std::unique_ptr<GPNode> clone() const override {
        return std::make_unique<ConstantNode>(value);
    }

    int size() const override { return 1; }
    int depth() const override { return 1; }

    void mutate(std::mt19937& gen, const GPConfig& config) override {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        value = std::tanh(dist(gen));  // Keep value in [-1, 1] range
    }
};

class OperatorNode : public GPNode {
private:
    std::string op_name;
    std::vector<std::unique_ptr<GPNode>> children;
    std::shared_ptr<GPOperations> ops;
    size_t node_id;
    static size_t next_node_id;

public:
    OperatorNode(std::string name,
                std::vector<std::unique_ptr<GPNode>> nodes,
                std::shared_ptr<GPOperations> operations)
        : op_name(std::move(name))
        , children(std::move(nodes))
        , ops(operations)
        , node_id(++next_node_id)
    {}

    std::vector<float> evaluate(const std::vector<float>& input) override {
        // Collect results from children
        std::vector<std::vector<float>> child_results;
        child_results.reserve(children.size());
        
        for (const auto& child : children) {
            if (child) {
                child_results.push_back(child->evaluate(input));
            }
        }

        // Use GPOperations to evaluate this node
        return ops->evaluate(op_name, child_results, node_id);
    }

    std::unique_ptr<GPNode> clone() const override {
        std::vector<std::unique_ptr<GPNode>> new_children;
        new_children.reserve(children.size());
        
        for (const auto& child : children) {
            if (child) {
                new_children.push_back(child->clone());
            }
        }
        
        return std::make_unique<OperatorNode>(op_name, std::move(new_children), ops);
    }

    int size() const override {
        int total = 1;
        for (const auto& child : children) {
            if (child) {
                total += child->size();
            }
        }
        return total;
    }

    int depth() const override {
        int max_depth = 0;
        for (const auto& child : children) {
            if (child) {
                max_depth = std::max(max_depth, child->depth());
            }
        }
        return max_depth + 1;
    }

    void mutate(std::mt19937& gen, const GPConfig& config) override {
        if (children.empty()) return;
        
        std::uniform_int_distribution<size_t> dist(0, children.size() - 1);
        size_t child_idx = dist(gen);
        
        if (children[child_idx]) {
            children[child_idx]->mutate(gen, config);
        }
    }
};

// Individual class representing a collection of trees
class Individual {
private:
    std::vector<std::unique_ptr<GPNode>> trees;
    float fitness;
    std::shared_ptr<GPOperations> ops;
    mutable std::mutex eval_mutex;

public:
    Individual(std::vector<std::unique_ptr<GPNode>> t, std::shared_ptr<GPOperations> operations) 
        : trees(std::move(t))
        , fitness(std::numeric_limits<float>::infinity())
        , ops(operations) {}

    // Deep copy constructor
    Individual(const Individual& other) 
        : fitness(other.fitness)
        , ops(other.ops)
    {
        std::lock_guard<std::mutex> lock(other.eval_mutex);
        trees.reserve(other.trees.size());
        for (const auto& tree : other.trees) {
            trees.push_back(tree->clone());
        }
    }

    // Move constructor
    Individual(Individual&& other) noexcept
        : trees(std::move(other.trees))
        , fitness(other.fitness)
        , ops(std::move(other.ops))
    {}

    // Copy assignment operator
    Individual& operator=(Individual other) {
        std::swap(trees, other.trees);
        fitness = other.fitness;
        ops = other.ops;
        return *this;
    }

    std::vector<float> evaluate(const std::vector<float>& input) const {
        std::lock_guard<std::mutex> lock(eval_mutex);
        
        ops->startNewEvaluation();
        std::vector<std::vector<float>> tree_results;
        tree_results.reserve(trees.size());
        
        for (const auto& tree : trees) {
            if (tree) {  // Add null check
                tree_results.push_back(tree->evaluate(input));
            }
        }
        
        if (tree_results.empty()) {
            return std::vector<float>{0.0f};  // Return safe default
        }
        
        std::vector<float> result(tree_results[0].size(), 0.0f);
        float scale = 1.0f / trees.size();
        
        for (const auto& tree_result : tree_results) {
            for (size_t i = 0; i < tree_result.size(); ++i) {
                result[i] += tree_result[i] * scale;
            }
        }
        
        return result;
    }

    void setFitness(float f) { fitness = f; }
    float getFitness() const { return fitness; }

    void mutate(std::mt19937& gen, const GPConfig& config) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& tree : trees) {
            if (dist(gen) < config.mutation_prob) {
                tree->mutate(gen, config);
            }
        }
    }

    int totalSize() const {
        int size = 0;
        for (const auto& tree : trees) {
            size += tree->size();
        }
        return size;
    }

    const std::unique_ptr<GPNode>& getTree(int index) const {
        return trees[index];
    }
};

class ContrastiveGP {
private:
    GPConfig config;
    std::shared_ptr<GPOperations> ops;
    std::vector<Individual> population;
    
    // Thread-safe RNG
    static thread_local std::mt19937 rng;

    std::mt19937& getGen() {
        return rng;  // Now returns reference to thread-local RNG
    }

    float calculateDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
        if (v1.empty() || v2.empty() || v1.size() != v2.size()) {
            std::cerr << "Warning: Empty or mismatched vectors in distance calculation. "
                    << "Sizes: " << v1.size() << " and " << v2.size() << std::endl;
            return std::numeric_limits<float>::max();
        }

        float sum = 0.0f;
        for (size_t i = 0; i < v1.size(); ++i) {
            float diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        
        // Normalize by vector dimension
        float distance = std::sqrt(sum) / std::sqrt(static_cast<float>(v1.size()));
        
        // Check for NaN or inf
        if (std::isnan(distance) || std::isinf(distance)) {
            std::cerr << "Warning: Invalid distance calculated" << std::endl;
            return std::numeric_limits<float>::max();
        }
        
        return distance;
    }

    float evaluateIndividual(const Individual& ind, const std::vector<DataPoint>& data) {
        float total_loss = 0.0f;
        float total_distance = 0.0f;
        int correct_predictions = 0;
        int total_pairs = 0;
        
        // Track separate losses for similar and dissimilar pairs
        float similar_loss = 0.0f;
        float dissimilar_loss = 0.0f;
        int similar_count = 0;
        int dissimilar_count = 0;
        
        for (size_t i = 0; i < data.size(); i += config.batch_size) {
            size_t batch_end = std::min(i + config.batch_size, data.size());
            std::vector<DataPoint> batch(data.begin() + i, data.begin() + batch_end);
            
            for (const auto& point : batch) {
                auto anchor_output = ind.evaluate(point.anchor);
                auto compare_output = ind.evaluate(point.compare);
                
                if (anchor_output.empty() || compare_output.empty()) {
                    std::cerr << "Warning: Empty output vectors in evaluation" << std::endl;
                    continue;
                }
                
                float distance = calculateDistance(anchor_output, compare_output);
                total_distance += distance;
                bool prediction = distance < config.distance_threshold;
                bool actual = point.label > 0.5f;
                
                if (prediction == actual) {
                    correct_predictions++;
                }
                
                // Contrastive loss calculation
                float loss;
                if (actual) {  // Similar pairs
                    loss = distance * distance;
                    similar_loss += loss;
                    similar_count++;
                } else {  // Dissimilar pairs
                    float margin_diff = std::max(0.0f, config.margin - distance);
                    loss = margin_diff * margin_diff;
                    dissimilar_loss += loss;
                    dissimilar_count++;
                }
                
                total_loss += loss;
                total_pairs++;
            }
        }
        
        if (total_pairs == 0) {
            std::cerr << "Warning: No pairs evaluated" << std::endl;
            return std::numeric_limits<float>::max();
        }
        
        float accuracy = static_cast<float>(correct_predictions) / total_pairs;
        float avg_loss = total_loss / total_pairs;
        float avg_distance = total_distance / total_pairs;
        
        // Calculate average losses for similar and dissimilar pairs
        float avg_similar_loss = similar_count > 0 ? similar_loss / similar_count : 0.0f;
        float avg_dissimilar_loss = dissimilar_count > 0 ? dissimilar_loss / dissimilar_count : 0.0f;
        
        // Combined fitness with better balancing
        float accuracy_term = (1.0f - accuracy) * config.fitness_alpha;
        float loss_term = avg_loss * config.loss_alpha;
        float complexity_term = config.parsimony_coeff * ind.totalSize();
        
        float fitness = accuracy_term + loss_term + complexity_term;
        
        // Check for invalid fitness
        if (std::isnan(fitness) || std::isinf(fitness)) {
            std::cerr << "Warning: Invalid fitness calculated" << std::endl;
            return std::numeric_limits<float>::max();
        }
        
        return fitness;
    }

    std::unique_ptr<GPNode> createRandomTree(int depth) {
        // Thread-local distributions
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        static thread_local std::uniform_int_distribution<int> op_dist(0, 6);
        static thread_local std::uniform_real_distribution<float> const_dist(-1.0f, 1.0f);

        // Early termination or leaf node creation
        if (depth <= 1 || (depth < config.max_tree_depth && prob_dist(rng) < 0.3)) {
            // Create leaf node (80% chance for feature node, 20% for constant)
            if (prob_dist(rng) < 0.8) {
                std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
                return std::make_unique<FeatureNode>(feature_dist(rng));
            } else {
                return std::make_unique<ConstantNode>(const_dist(rng));
            }
        }

        // Create children for operator node
        std::vector<std::unique_ptr<GPNode>> children;
        children.push_back(createRandomTree(depth - 1));
        children.push_back(createRandomTree(depth - 1));

        // Select operation type
        int op_choice = op_dist(rng);
        std::string op_name;

        switch (op_choice) {
            case 0:
                op_name = "add";
                break;
            case 1:
                op_name = "sub";
                break;
            case 2:
                op_name = "div";
                break;
            case 3:
                op_name = "sin";
                break;
            case 4:
                op_name = "cos";
                break;
            case 5:
                op_name = "neg";
                break;
            default:
                op_name = "mul";
                break;
        }

        // Create operator node with operation name, children, and ops pointer
        return std::make_unique<OperatorNode>(
            op_name,
            std::move(children),
            ops
        );
        }

    Individual createRandomIndividual() {
        std::vector<std::unique_ptr<GPNode>> trees;
        for (int i = 0; i < config.num_trees; ++i) {
            trees.push_back(createRandomTree(config.max_tree_depth));
        }
        return Individual(std::move(trees), ops);  // Pass ops to Individual
    }

    std::vector<float> generateRandomVector(std::mt19937& gen, float min = -1.0f, float max = 1.0f) {
        std::vector<float> vec(1023);
        std::uniform_real_distribution<float> dist(min, max);
        
        for (auto& val : vec) {
            val = dist(gen);
        }
        return vec;
    }
    
    // Helper method to add noise to a vector
    std::vector<float> addNoise(const std::vector<float>& vec, float noise_level, std::mt19937& gen) {
        std::vector<float> noisy_vec = vec;
        std::normal_distribution<float> noise(0.0f, noise_level);
        
        for (auto& val : noisy_vec) {
            val += noise(gen);
            // Clamp values to reasonable range
            val = std::max(-1.0f, std::min(1.0f, val));
        }
        return noisy_vec;
    }

    float calculateAccuracy(const Individual& ind, const std::vector<DataPoint>& data) {
        int correct_predictions = 0;
        int total_pairs = 0;
        
        // Keep track of distance statistics
        float min_distance = std::numeric_limits<float>::max();
        float max_distance = std::numeric_limits<float>::min();
        float avg_distance = 0.0f;
        float similar_pair_avg_dist = 0.0f;
        float dissimilar_pair_avg_dist = 0.0f;
        int similar_pairs = 0;
        int dissimilar_pairs = 0;
        
        for (size_t i = 0; i < data.size(); i += config.batch_size) {
            size_t batch_end = std::min(i + config.batch_size, data.size());
            std::vector<DataPoint> batch(data.begin() + i, data.begin() + batch_end);
            
            for (const auto& point : batch) {
                auto anchor_output = ind.evaluate(point.anchor);
                auto compare_output = ind.evaluate(point.compare);
                
                float distance = calculateDistance(anchor_output, compare_output);
                bool prediction = distance < config.distance_threshold;
                bool actual = point.label > 0.5f;
                
                // Update distance statistics
                min_distance = std::min(min_distance, distance);
                max_distance = std::max(max_distance, distance);
                avg_distance += distance;
                
                if (actual) {
                    similar_pair_avg_dist += distance;
                    similar_pairs++;
                } else {
                    dissimilar_pair_avg_dist += distance;
                    dissimilar_pairs++;
                }
                
                if (prediction == actual) {
                    correct_predictions++;
                }
                total_pairs++;
            }
        }
        
        // Print distance statistics
        if (total_pairs > 0) {
            avg_distance /= total_pairs;
            if (similar_pairs > 0) similar_pair_avg_dist /= similar_pairs;
            if (dissimilar_pairs > 0) dissimilar_pair_avg_dist /= dissimilar_pairs;
            
            std::cout << "\n  Distance Statistics:"
                    << "\n    Min Distance: " << min_distance
                    << "\n    Max Distance: " << max_distance
                    << "\n    Avg Distance: " << avg_distance
                    << "\n    Similar Pairs Avg Distance: " << similar_pair_avg_dist
                    << "\n    Dissimilar Pairs Avg Distance: " << dissimilar_pair_avg_dist
                    << "\n    Distance Threshold: " << config.distance_threshold << std::endl;
        }
        
        return static_cast<float>(correct_predictions) / total_pairs;
    }

public:
    ContrastiveGP(const GPConfig& cfg) : config(cfg) {
        // Initialize operations as shared_ptr
        ops = std::make_shared<GPOperations>(
            cfg.dropout_prob, cfg.bn_momentum, cfg.bn_epsilon);
        
        // Initialize population
        population.reserve(config.population_size);
        for (int i = 0; i < config.population_size; ++i) {
            population.push_back(createRandomIndividual());
        }
    }

    void train(const std::vector<DataPoint>& trainData,
          const std::vector<DataPoint>& valData) {
        float best_fitness = std::numeric_limits<float>::max();
        int generations_without_improvement = 0;

        for (int generation = 0; generation < config.generations; ++generation) {
            std::cout << "\nStarting generation " << generation << std::endl;
            std::cout << "Population size: " << population.size() << std::endl;
            
            ops->setTraining(true);

            // Create batch indices for parallel processing
            std::vector<std::pair<size_t, size_t>> batch_indices;
            const size_t num_workers = std::min((size_t)config.num_workers, population.size());
            const size_t batch_size = (population.size() + num_workers - 1) / num_workers;
            
            for (size_t i = 0; i < population.size(); i += batch_size) {
                size_t end = std::min(i + batch_size, population.size());
                batch_indices.emplace_back(i, end);
            }

            // Evaluate batches in parallel
            std::vector<std::future<void>> futures;
            std::vector<float> fitnesses(population.size(), std::numeric_limits<float>::max());
            std::mutex mtx;

            for (const auto& [start, end] : batch_indices) {
                futures.push_back(std::async(std::launch::async, [&, start, end]() {
                    try {
                        std::vector<std::pair<size_t, float>> local_results;
                        local_results.reserve(end - start);

                        for (size_t i = start; i < end; ++i) {
                            float fitness = evaluateIndividual(population[i], trainData);
                            local_results.emplace_back(i, fitness);
                        }

                        // Update results with mutex protection
                        std::lock_guard<std::mutex> lock(mtx);
                        for (const auto& [idx, fitness] : local_results) {
                            fitnesses[idx] = fitness;
                            population[idx].setFitness(fitness);
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Worker error: " << e.what() << std::endl;
                    }
                }));
            }

            // Wait for all evaluations to complete
            for (auto& future : futures) {
                future.wait();
            }

            // Find best individual and calculate statistics
            float gen_best_fitness = std::numeric_limits<float>::max();
            float gen_avg_fitness = 0.0f;
            size_t best_idx = 0;

            for (size_t i = 0; i < population.size(); ++i) {
                float fitness = fitnesses[i];
                gen_avg_fitness += fitness;
                if (fitness < gen_best_fitness) {
                    gen_best_fitness = fitness;
                    best_idx = i;
                }
            }

            gen_avg_fitness /= population.size();

            // Calculate accuracies
            float train_accuracy = calculateAccuracy(population[best_idx], trainData);
            float val_accuracy = calculateAccuracy(population[best_idx], valData);

            // Print progress
            std::cout << "Generation " << generation 
                    << "\n  Best Fitness: " << gen_best_fitness
                    << "\n  Avg Fitness: " << gen_avg_fitness
                    << "\n  Training Accuracy: " << train_accuracy * 100.0f << "%"
                    << "\n  Validation Accuracy: " << val_accuracy * 100.0f << "%"
                    << std::endl;

            std::cout << "I get here" << std::endl;

            // Check for improvement
            if (gen_best_fitness < best_fitness) {
                best_fitness = gen_best_fitness;
                generations_without_improvement = 0;
            } else {
                generations_without_improvement++;
            }

            std::cout << "I get here II" << std::endl;


            if (generations_without_improvement > 20) {
                std::cout << "Early stopping due to lack of improvement" << std::endl;
                break;
            }

            std::cout << "I get here III" << std::endl;

            // Create new population
            std::vector<Individual> new_population;
            new_population.reserve(config.population_size);

            // Sort current population by fitness
            std::vector<size_t> indices(population.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                    [&](size_t a, size_t b) { return fitnesses[a] < fitnesses[b]; });

            std::cout << "I get here IV" << std::endl;

            // Elitism
            for (int i = 0; i < config.elite_size && i < population.size(); ++i) {
                new_population.push_back(Individual(population[indices[i]]));
            }

            std::cout << "I get here V" << std::endl;

            // DEBUG - segmenation fault here on second iteration.

            // Generate remaining individuals
            auto& rng = getGen();
            std::uniform_int_distribution<size_t> parent_dist(0, indices.size() - 1);
            std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

            while (new_population.size() < config.population_size) {
                // Tournament selection
                size_t parent1_idx = indices[parent_dist(rng)];
                size_t parent2_idx = indices[parent_dist(rng)];
                
                // Crossover
                Individual offspring = crossover(population[parent1_idx], 
                                            population[parent2_idx]);
                
                // Mutation
                if (prob_dist(rng) < config.mutation_prob) {
                    offspring.mutate(rng, config);
                }
                
                new_population.push_back(std::move(offspring));
            }

            std::cout << "I get here VI" << std::endl;

            // Replace population
            population = std::move(new_population);
        }
    }

    Individual selectParent() {
        std::vector<Individual> tournament;
        auto& local_gen = getGen();  // Get thread-local RNG
        
        for (int i = 0; i < config.tournament_size; ++i) {
            std::uniform_int_distribution<int> dist(0, population.size() - 1);
            tournament.push_back(population[dist(local_gen)]);
        }
        
        return *std::min_element(tournament.begin(), tournament.end(),
                               [](const auto& a, const auto& b) {
                                   return a.getFitness() < b.getFitness();
                               });
    }

    Individual crossover(const Individual& parent1, const Individual& parent2) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        std::vector<std::unique_ptr<GPNode>> offspring_trees;
        offspring_trees.reserve(config.num_trees);

        if (dist(rng) < config.crossover_prob) {
            for (int i = 0; i < config.num_trees; ++i) {
                if (dist(rng) < 0.5f) {
                    offspring_trees.push_back(parent1.getTree(i)->clone());
                } else {
                    offspring_trees.push_back(parent2.getTree(i)->clone());
                }
            }
        }

        // Create new Individual with the trees and shared operations pointer
        return Individual(std::move(offspring_trees), this->ops);
    }

    // Method to generate synthetic data pairs
    std::pair<std::vector<DataPoint>, std::vector<DataPoint>> 
    generateSyntheticData(int num_train_pairs, int num_val_pairs, float noise_level = 0.1f) {
        std::vector<DataPoint> trainData;
        std::vector<DataPoint> valData;
        
        // Separate random generator for data generation
        std::mt19937 data_gen(std::random_device{}());
        std::uniform_real_distribution<float> coin_flip(0.0f, 1.0f);
        
        // Generate training data
        for (int i = 0; i < num_train_pairs; ++i) {
            bool is_similar = coin_flip(data_gen) < 0.5f;  // 50% similar, 50% dissimilar pairs
            
            if (is_similar) {
                // Generate similar pair by adding noise to the same base vector
                auto base_vec = generateRandomVector(data_gen);
                auto noisy_vec = addNoise(base_vec, noise_level, data_gen);
                trainData.emplace_back(base_vec, noisy_vec, 1.0f);
            } else {
                // Generate dissimilar pair using two different random vectors
                auto vec1 = generateRandomVector(data_gen);
                auto vec2 = generateRandomVector(data_gen);
                trainData.emplace_back(vec1, vec2, 0.0f);
            }
        }
        
        // Generate validation data
        for (int i = 0; i < num_val_pairs; ++i) {
            bool is_similar = coin_flip(data_gen) < 0.5f;
            
            if (is_similar) {
                auto base_vec = generateRandomVector(data_gen);
                auto noisy_vec = addNoise(base_vec, noise_level, data_gen);
                valData.emplace_back(base_vec, noisy_vec, 1.0f);
            } else {
                auto vec1 = generateRandomVector(data_gen);
                auto vec2 = generateRandomVector(data_gen);
                valData.emplace_back(vec1, vec2, 0.0f);
            }
        }
        
        return {trainData, valData};
    }
};

size_t OperatorNode::next_node_id = 0;
// Initialize static member
thread_local std::mt19937 ContrastiveGP::rng{std::random_device{}()};
thread_local std::mt19937 GPOperations::rng{std::random_device{}()};

// Example usage:
int main() {
    // Initialize configuration
    GPConfig config;

    std::cout << "Using " << config.num_workers << " worker threads" << std::endl;
    
    // Create and initialize model
    ContrastiveGP model(config);
    
    // Read and process data
    ExcelProcessor processor;
    try {

        std::cout << "Reading the Excel file" << std::endl;
        auto allData = processor.readExcel("/home/woodj/Desktop/fishy-business/data/REIMS.xlsx", "All data no QC filtering");
        auto [trainData, valData] = processor.splitTrainVal(allData);
        
        // Train 
        std::cout << "Training" << std::endl;
        model.train(trainData, valData);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}