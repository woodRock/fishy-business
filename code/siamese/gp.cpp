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
    int num_trees = 20;
    int population_size = 100;
    int generations = 100;
    int elite_size = 10;
    float crossover_prob = 0.8f;
    float mutation_prob = 0.3f;
    int tournament_size = 7;
    float distance_threshold = 0.8f;
    float margin = 1.0f;
    float fitness_alpha = 0.8f;
    float loss_alpha = 0.2f;
    float balance_alpha = 0.001f;
    int max_tree_depth = 6;
    float parsimony_coeff = 0.001f;
    int batch_size = 128;
    int num_workers = 15;
    float dropout_prob = 0.0f;
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
private:
    float dropout_prob;
    bool training;
    std::map<std::string, std::unique_ptr<BatchNorm>> batch_norms;
    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<float> dist{0.0f, 1.0f};

public:
    GPOperations(float dp = 0.2f, float momentum = 0.01f, float epsilon = 0.1f)  // Modified parameters
        : dropout_prob(dp), training(true) {
        std::vector<std::string> ops = {"add", "sub", "mul", "div", "sin", "cos", "neg"};
        for (const auto& op : ops) {
            batch_norms[op] = std::make_unique<BatchNorm>(momentum, epsilon);
        }
    }


    void setTraining(bool mode) {
        training = mode;
        for (auto& [_, bn] : batch_norms) {
            bn->setTraining(mode);
        }
    }

    std::vector<float> maybeDropout(const std::vector<float>& x) {
        if (training && dist(gen) < dropout_prob) {
            return std::vector<float>(x.size(), 0.0f);
        }
        return x;
    }

    // Operation implementations
    std::vector<float> add(const std::vector<float>& x, const std::vector<float>& y) {
        auto result = vec_ops::add(x, y);
        
        // Apply batch norm with debugging
        result = batch_norms["add"]->operator()(result);
        
        return result;  // Temporarily disable dropout for debugging
    }

    std::vector<float> sub(const std::vector<float>& x, const std::vector<float>& y) {
        auto result = vec_ops::subtract(x, y);
        result = batch_norms["sub"]->operator()(result);
        return maybeDropout(result);
    }

    std::vector<float> mul(const std::vector<float>& x, const std::vector<float>& y) {
        auto result = vec_ops::multiply(x, y);
        result = batch_norms["mul"]->operator()(result);
        return maybeDropout(result);
    }

    std::vector<float> protectedDiv(const std::vector<float>& x, const std::vector<float>& y) {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::abs(y[i]) < 1e-10f ? 1.0f : x[i] / y[i];
        }
        result = batch_norms["div"]->operator()(result);
        return maybeDropout(result);
    }

    std::vector<float> sin(const std::vector<float>& x) {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::sin(x[i]);
        }
        result = batch_norms["sin"]->operator()(result);
        return maybeDropout(result);
    }

    std::vector<float> cos(const std::vector<float>& x) {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::cos(x[i]);
        }
        result = batch_norms["cos"]->operator()(result);
        return maybeDropout(result);
    }

    std::vector<float> neg(const std::vector<float>& x) {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = -x[i];
        }
        result = batch_norms["neg"]->operator()(result);
        return maybeDropout(result);
    }

    std::vector<float> sumReduce(const std::vector<std::vector<float>>& inputs) {
        if (inputs.empty()) return std::vector<float>();
        std::vector<float> result = inputs[0];
        for (size_t i = 1; i < inputs.size(); ++i) {
            result = add(result, inputs[i]);
        }
        return result;
    }

    std::vector<float> meanReduce(const std::vector<std::vector<float>>& inputs) {
        if (inputs.empty()) {
            std::cerr << "Warning: Empty inputs in meanReduce" << std::endl;
            return std::vector<float>();
        }
        
        auto sum = sumReduce(inputs);
        if (!sum.empty()) {
            for (float& val : sum) {
                val /= inputs.size();
                // Check for NaN or inf
                if (std::isnan(val) || std::isinf(val)) {
                    std::cerr << "Warning: Invalid value in meanReduce output" << std::endl;
                }
            }
        }
        return sum;
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
protected:
    std::string op_name;
    std::vector<std::unique_ptr<GPNode>> children;
    std::function<std::vector<float>(const std::vector<std::vector<float>>&)> operation;

public:
    OperatorNode(std::string name, 
                std::function<std::vector<float>(const std::vector<std::vector<float>>&)> op,
                std::vector<std::unique_ptr<GPNode>> nodes)
        : op_name(std::move(name))
        , operation(std::move(op))
        , children(std::move(nodes)) {}

    std::vector<float> evaluate(const std::vector<float>& input) override {        
        std::vector<std::vector<float>> child_results;
        for (const auto& child : children) {
            auto result = child->evaluate(input);
            child_results.push_back(std::move(result));
        }
        
        auto result = operation(child_results);
        
        return result;
    }

    std::unique_ptr<GPNode> clone() const override {
        std::vector<std::unique_ptr<GPNode>> new_children;
        for (const auto& child : children) {
            new_children.push_back(child->clone());
        }
        return std::make_unique<OperatorNode>(op_name, operation, std::move(new_children));
    }

    int size() const override {
        int total = 1;
        for (const auto& child : children) {
            total += child->size();
        }
        return total;
    }

    int depth() const override {
        int max_depth = 0;
        for (const auto& child : children) {
            max_depth = std::max(max_depth, child->depth());
        }
        return max_depth + 1;
    }

    void mutate(std::mt19937& gen, const GPConfig& config) override {
        std::uniform_int_distribution<int> dist(0, children.size() - 1);
        int child_idx = dist(gen);
        children[child_idx]->mutate(gen, config);
    }
};

// Individual class representing a collection of trees
class Individual {
private:
    std::vector<std::unique_ptr<GPNode>> trees;
    float fitness;

public:
    Individual(std::vector<std::unique_ptr<GPNode>> t) 
        : trees(std::move(t)), fitness(std::numeric_limits<float>::infinity()) {}

    Individual(const Individual& other) {
        trees.reserve(other.trees.size());
        for (const auto& tree : other.trees) {
            trees.push_back(tree->clone());
        }
        fitness = other.fitness;
    }

    Individual& operator=(Individual other) {
        std::swap(trees, other.trees);
        fitness = other.fitness;
        return *this;
    }

    void setFitness(float f) { fitness = f; }
    float getFitness() const { return fitness; }

    std::vector<float> evaluate(const std::vector<float>& input) const {        
        std::vector<float> result;
        int tree_idx = 0;
        for (const auto& tree : trees) {
            auto tree_result = tree->evaluate(input);
            result.insert(result.end(), tree_result.begin(), tree_result.end());
            tree_idx++;
        }
        
        return result;
    }


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
    std::unique_ptr<GPOperations> ops;
    std::vector<Individual> population;
    std::mt19937 gen{std::random_device{}()};

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
        
        // Debug output
        // std::cout << "\n  Evaluation Stats:"
        //         << "\n    Accuracy: " << (accuracy * 100.0f) << "%"
        //         << "\n    Average Loss: " << avg_loss
        //         << "\n    Average Distance: " << avg_distance
        //         << "\n    Similar Pairs Loss: " << avg_similar_loss
        //         << "\n    Dissimilar Pairs Loss: " << avg_dissimilar_loss
        //         << "\n    Total Pairs: " << total_pairs
        //         << "\n    Similar/Dissimilar Ratio: " 
        //         << similar_count << "/" << dissimilar_count << std::endl;
        
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
        if (depth <= 1 || (depth < config.max_tree_depth && 
            std::uniform_real_distribution<float>(0, 1)(gen) < 0.3)) {
            // Create leaf node
            if (std::uniform_real_distribution<float>(0, 1)(gen) < 0.8) {
                int feature_idx = std::uniform_int_distribution<int>(0, config.n_features - 1)(gen);
                return std::make_unique<FeatureNode>(feature_idx);
            } else {
                float value = std::uniform_real_distribution<float>(-1, 1)(gen);
                return std::make_unique<ConstantNode>(value);
            }
        }

        // Create operator node
        std::vector<std::unique_ptr<GPNode>> children;
        children.push_back(createRandomTree(depth - 1));
        children.push_back(createRandomTree(depth - 1));
        
        // Choose random operation
        std::uniform_int_distribution<int> op_dist(0, 6);
        int op_choice = op_dist(gen);
        std::string op_name;
        
        std::function<std::vector<float>(const std::vector<std::vector<float>>&)> operation;
        
        switch (op_choice) {
            case 0:
                op_name = "add";
                operation = [this](const auto& inputs) { return ops->add(inputs[0], inputs[1]); };
                break;
            case 1:
                op_name = "sub";
                operation = [this](const auto& inputs) { return ops->sub(inputs[0], inputs[1]); };
                break;
            case 2:
                op_name = "div";
                operation = [this](const auto& inputs) { return ops->protectedDiv(inputs[0], inputs[1]); };
                break;
            case 3:
                op_name = "sin";
                operation = [this](const auto& inputs) { return ops->sin(inputs[0]); };
                break;
            case 4:
                op_name = "cos";
                operation = [this](const auto& inputs) { return ops->cos(inputs[0]); };
                break;
            case 5:
                op_name = "neg";
                operation = [this](const auto& inputs) { return ops->neg(inputs[0]); };
                break;
            default:
                op_name = "mul";
                operation = [this](const auto& inputs) { return ops->mul(inputs[0], inputs[1]); };
                break;
        }
        
        return std::make_unique<OperatorNode>(op_name, operation, std::move(children));
    }

    Individual createRandomIndividual() {
        std::vector<std::unique_ptr<GPNode>> trees;
        for (int i = 0; i < config.num_trees; ++i) {
            trees.push_back(createRandomTree(config.max_tree_depth));
        }
        return Individual(std::move(trees));
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

public:
    ContrastiveGP(const GPConfig& cfg) : config(cfg) {
        ops = std::make_unique<GPOperations>(
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
        
        // Define calculateAccuracy lambda inside train
        auto calculateAccuracy = [this](const Individual& ind, const std::vector<DataPoint>& data) {
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
            
            return static_cast<float>(correct_predictions) / total_pairs;
        };

        for (int generation = 0; generation < config.generations; ++generation) {
            // Set training mode for batch normalization
            ops->setTraining(true);
            
            // Evaluate population
            float gen_best_fitness = std::numeric_limits<float>::max();
            float gen_avg_fitness = 0.0f;
            Individual* best_individual = nullptr;
            
            for (auto& ind : population) {
                float fitness = evaluateIndividual(ind, trainData);
                ind.setFitness(fitness);
                if (fitness < gen_best_fitness) {
                    gen_best_fitness = fitness;
                    best_individual = &ind;
                }
                gen_avg_fitness += fitness;
            }
            gen_avg_fitness /= population.size();

            // Calculate and print accuracies for best individual
            float train_accuracy = calculateAccuracy(*best_individual, trainData);
            float val_accuracy = calculateAccuracy(*best_individual, valData);

            // Check improvement
            if (gen_best_fitness < best_fitness) {
                best_fitness = gen_best_fitness;
                generations_without_improvement = 0;
            } else {
                generations_without_improvement++;
            }

            // Print detailed progress
            std::cout << "Generation " << generation 
                    << "\n  Best Fitness: " << gen_best_fitness
                    << "\n  Avg Fitness: " << gen_avg_fitness
                    << "\n  Training Accuracy: " << train_accuracy * 100.0f << "%"
                    << "\n  Validation Accuracy: " << val_accuracy * 100.0f << "%"
                    << "\n  Generations without improvement: " 
                    << generations_without_improvement << std::endl;

            // Early stopping if no improvement for many generations
            if (generations_without_improvement > 20) {
                std::cout << "Early stopping due to lack of improvement" << std::endl;
                break;
            }

            // Sort population by fitness
            std::sort(population.begin(), population.end(),
                    [](const auto& a, const auto& b) {
                        return a.getFitness() < b.getFitness();
                    });

            // Create new population
            std::vector<Individual> new_population;
            
            // Elitism - keep best individuals
            for (int i = 0; i < config.elite_size; ++i) {
                new_population.push_back(population[i]);
            }
            
            // Breed new individuals
            while (new_population.size() < config.population_size) {
                // Tournament selection
                auto parent1 = selectParent();
                auto parent2 = selectParent();
                
                // Create offspring through crossover and mutation
                Individual offspring = crossover(parent1, parent2);
                offspring.mutate(gen, config);
                
                new_population.push_back(std::move(offspring));
            }
            
            // Replace old population
            population = std::move(new_population);
        }
    }

    Individual selectParent() {
        std::vector<Individual> tournament;
        for (int i = 0; i < config.tournament_size; ++i) {
            std::uniform_int_distribution<int> dist(0, population.size() - 1);
            tournament.push_back(population[dist(gen)]);
        }
        
        return *std::min_element(tournament.begin(), tournament.end(),
                               [](const auto& a, const auto& b) {
                                   return a.getFitness() < b.getFitness();
                               });
    }

    Individual crossover(const Individual& parent1, const Individual& parent2) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        // Create new trees vector for offspring
        std::vector<std::unique_ptr<GPNode>> offspring_trees;
        offspring_trees.reserve(config.num_trees);

        // Perform crossover with probability config.crossover_prob
        if (dist(gen) < config.crossover_prob) {
            // Mix trees from both parents with random selection
            for (int i = 0; i < config.num_trees; ++i) {
                if (dist(gen) < 0.5f) {
                    offspring_trees.push_back(parent1.getTree(i)->clone());
                } else {
                    offspring_trees.push_back(parent2.getTree(i)->clone());
                }
            }
        } else {
            // Clone the better parent
            const Individual& better_parent = 
                (parent1.getFitness() < parent2.getFitness()) ? parent1 : parent2;
            for (int i = 0; i < config.num_trees; ++i) {
                offspring_trees.push_back(better_parent.getTree(i)->clone());
            }
        }

        return Individual(std::move(offspring_trees));
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


// Example usage:
int main() {
    // Initialize configuration
    GPConfig config;
    
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