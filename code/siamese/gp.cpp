// ContrastiveGP.hpp
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
#include <chrono>
#include <fstream>
#include <optional>

// Configuration struct
struct GPConfig {
    int n_features = 2080;
    int num_trees = 10;                   // Increased from 20 to 30 for better ensemble
    int population_size = 100;            // Doubled population size for more diversity
    int generations = 100;                // Doubled to allow more evolution time
    int elite_size = 10;                  // Increased to preserve good solutions
    float crossover_prob = 0.85f;         // Slightly increased for more genetic material exchange
    float mutation_prob = 0.25f;          // Increased for better exploration
    int tournament_size = 7;              // Increased for stronger selection pressure
    float distance_threshold = 0.4f;      // Reduced to be more selective
    float margin = 1.5f;                  // Increased margin for better separation
    float fitness_alpha = 0.7f;           // Reduced to balance accuracy vs. complexity
    float loss_alpha = 0.3f;              // Increased to focus more on loss reduction
    float parsimony_coeff = 0.0005f;      // Reduced to allow more complex solutions
    int max_tree_depth = 6;               // Increased for more expressive trees
    int min_tree_depth = 2;               // Kept the same
    int batch_size = 32;                  // Reduced for more frequent updates
    int num_workers = std::thread::hardware_concurrency();
    float dropout_prob = 0.15f;           // Increased for stronger regularization
    float bn_momentum = 0.01f;            // Reduced  faster adaptation
    float bn_epsilon = 1e-6f;             // Reduced for more precise normalization
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

    std::vector<DataPoint> generatePairs(const std::vector<DataPoint>& instances, size_t pairs_per_sample = 50) {
        // Initialize random number generators
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        // Create class indices map
        std::unordered_map<std::string, std::vector<size_t>> class_indices;
        for (size_t idx = 0; idx < instances.size(); ++idx) {
            const auto& label = instances[idx].label;
            if (label > 0.5f) {
                class_indices["1"].push_back(idx);
            } else {
                class_indices["0"].push_back(idx);
            }
        }
        
        // Generate pairs
        std::vector<DataPoint> pairs;
        const size_t expected_pairs = instances.size() * pairs_per_sample;
        pairs.reserve(expected_pairs);
        
        for (size_t idx1 = 0; idx1 < instances.size(); ++idx1) {
            const auto& feat1 = instances[idx1];
            const std::string label1_key = feat1.label > 0.5f ? "1" : "0";
            
            for (size_t p = 0; p < pairs_per_sample; ++p) {
                size_t idx2;
                
                if (dist(gen) < 0.5f) {
                    // Try to get same class sample
                    const auto& same_class_indices = class_indices[label1_key];
                    if (same_class_indices.size() > 1) {
                        // Filter out the current index
                        std::vector<size_t> valid_indices;
                        std::copy_if(same_class_indices.begin(), 
                                same_class_indices.end(),
                                std::back_inserter(valid_indices),
                                [idx1](size_t idx) { return idx != idx1; });
                        
                        std::uniform_int_distribution<size_t> index_dist(0, valid_indices.size() - 1);
                        idx2 = valid_indices[index_dist(gen)];
                    } else {
                        std::uniform_int_distribution<size_t> index_dist(0, instances.size() - 1);
                        idx2 = index_dist(gen);
                    }
                } else {
                    // Random sample from different class
                    std::uniform_int_distribution<size_t> index_dist(0, instances.size() - 1);
                    idx2 = index_dist(gen);
                }
                
                const auto& feat2 = instances[idx2];
                float pair_label = (feat1.label > 0.5f) == (feat2.label > 0.5f) ? 1.0f : 0.0f;
                
                // Create pair using DataPoint constructor
                pairs.emplace_back(feat1.anchor, feat2.anchor, pair_label);
            }
        }
        
        // Shuffle the pairs
        std::shuffle(pairs.begin(), pairs.end(), gen);
        
        // Log statistics
        size_t positive_pairs = std::count_if(pairs.begin(), pairs.end(),
            [](const DataPoint& p) { return p.label > 0.5f; });
        size_t negative_pairs = pairs.size() - positive_pairs;
        
        std::cout << "\nPair Generation Statistics:"
                << "\nTotal pairs generated: " << pairs.size()
                << "\nPositive pairs: " << positive_pairs 
                << " (" << (100.0f * positive_pairs / pairs.size()) << "%)"
                << "\nNegative pairs: " << negative_pairs
                << " (" << (100.0f * negative_pairs / pairs.size()) << "%)"
                << "\nAverage pairs per instance: " 
                << (static_cast<float>(pairs.size()) / instances.size()) 
                << std::endl;
        
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

            // Create temporary vector for instances
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

            // Convert instances to DataPoints
            std::vector<DataPoint> dataPoints;
            dataPoints.reserve(instances.size());
            
            // Create label encoder
            std::unordered_map<std::string, float> labelEncoder;
            float currentLabel = 0.0f;
            for (const auto& instance : instances) {
                if (labelEncoder.find(instance.label) == labelEncoder.end()) {
                    labelEncoder[instance.label] = currentLabel++;
                }
            }
            
            // Convert instances to DataPoints with encoded labels
            for (const auto& instance : instances) {
                float encodedLabel = labelEncoder[instance.label];
                dataPoints.emplace_back(instance.features, encodedLabel);
            }

            // Shuffle dataPoints before generating pairs
            std::random_device rd;
            std::mt19937 shuffler(rd());
            std::shuffle(dataPoints.begin(), dataPoints.end(), shuffler);
            
            // Generate pairs with shuffled DataPoints
            std::cout << "Generating pairs from shuffled instances..." << std::endl;
            auto pairs = generatePairs(dataPoints);
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
    std::pair<std::vector<DataPoint>, std::vector<DataPoint>> splitTrainVal(const std::vector<DataPoint>& data, float valRatio = 0.2) {
        std::vector<DataPoint> trainData;
        std::vector<DataPoint> valData;
        
        // First separate positive and negative pairs
        std::vector<const DataPoint*> positivePairs;
        std::vector<const DataPoint*> negativePairs;
        
        for (const auto& point : data) {
            if (point.label > 0.5f) {
                positivePairs.push_back(&point);
            } else {
                negativePairs.push_back(&point);
            }
        }
        
        // Calculate validation set sizes
        size_t numPosVal = static_cast<size_t>(positivePairs.size() * valRatio);
        size_t numNegVal = static_cast<size_t>(negativePairs.size() * valRatio);
        
        // Create random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Shuffle both positive and negative pairs
        std::shuffle(positivePairs.begin(), positivePairs.end(), gen);
        std::shuffle(negativePairs.begin(), negativePairs.end(), gen);
        
        // Reserve space for efficiency
        trainData.reserve(data.size() - numPosVal - numNegVal);
        valData.reserve(numPosVal + numNegVal);
        
        // Split positive pairs
        for (size_t i = 0; i < positivePairs.size(); ++i) {
            if (i < numPosVal) {
                valData.push_back(*positivePairs[i]);
            } else {
                trainData.push_back(*positivePairs[i]);
            }
        }
        
        // Split negative pairs
        for (size_t i = 0; i < negativePairs.size(); ++i) {
            if (i < numNegVal) {
                valData.push_back(*negativePairs[i]);
            } else {
                trainData.push_back(*negativePairs[i]);
            }
        }
        
        // Calculate normalization parameters from training data only
        std::vector<float> means, stds;
        if (!trainData.empty() && !trainData[0].anchor.empty()) {
            size_t numFeatures = trainData[0].anchor.size();
            std::vector<float> means(numFeatures, 0.0f);
            std::vector<float> stds(numFeatures, 0.0f);
            
            // First pass: compute means
            size_t totalVectors = 0;
            for (const auto& point : trainData) {
                for (size_t i = 0; i < numFeatures; ++i) {
                    means[i] += point.anchor[i];
                    means[i] += point.compare[i];
                }
                totalVectors += 2;  // Count both anchor and compare
            }
            
            // Finalize means
            for (auto& mean : means) {
                mean /= totalVectors;
            }
            
            // Second pass: compute standard deviations
            for (const auto& point : trainData) {
                for (size_t i = 0; i < numFeatures; ++i) {
                    float diff_anchor = point.anchor[i] - means[i];
                    float diff_compare = point.compare[i] - means[i];
                    stds[i] += diff_anchor * diff_anchor;
                    stds[i] += diff_compare * diff_compare;
                }
            }
            
            // Finalize standard deviations
            for (auto& std : stds) {
                std = std::sqrt(std / totalVectors);
                if (std < 1e-10f) std = 1.0f;  // Prevent division by zero
            }

            // Print pre-normalization statistics
            std::cout << "\nPre-normalization statistics:"
                    << "\n  Mean range: [" << *std::min_element(means.begin(), means.end())
                    << ", " << *std::max_element(means.begin(), means.end()) << "]"
                    << "\n  Std range: [" << *std::min_element(stds.begin(), stds.end())
                    << ", " << *std::max_element(stds.begin(), stds.end()) << "]" << std::endl;
            
            // Apply normalization to training data
            for (auto& point : trainData) {
                std::vector<float> normalized_anchor(numFeatures);
                std::vector<float> normalized_compare(numFeatures);
                
                for (size_t i = 0; i < numFeatures; ++i) {
                    normalized_anchor[i] = (point.anchor[i] - means[i]) / stds[i];
                    normalized_compare[i] = (point.compare[i] - means[i]) / stds[i];
                }
                point.anchor = std::move(normalized_anchor);
                point.compare = std::move(normalized_compare);
            }
            
            // Apply same normalization to validation data
            for (auto& point : valData) {
                std::vector<float> normalized_anchor(numFeatures);
                std::vector<float> normalized_compare(numFeatures);
                
                for (size_t i = 0; i < numFeatures; ++i) {
                    normalized_anchor[i] = (point.anchor[i] - means[i]) / stds[i];
                    normalized_compare[i] = (point.compare[i] - means[i]) / stds[i];
                }
                point.anchor = std::move(normalized_anchor);
                point.compare = std::move(normalized_compare);
            }

            // Verify normalization by computing statistics after
            std::vector<float> post_means(numFeatures, 0.0f);
            std::vector<float> post_stds(numFeatures, 0.0f);
            
            // Compute post-normalization statistics on training data
            totalVectors = 0;
            for (const auto& point : trainData) {
                for (size_t i = 0; i < numFeatures; ++i) {
                    post_means[i] += point.anchor[i];
                    post_means[i] += point.compare[i];
                }
                totalVectors += 2;
            }
            
            for (auto& mean : post_means) {
                mean /= totalVectors;
            }
            
            for (const auto& point : trainData) {
                for (size_t i = 0; i < numFeatures; ++i) {
                    float diff_anchor = point.anchor[i] - post_means[i];
                    float diff_compare = point.compare[i] - post_means[i];
                    post_stds[i] += diff_anchor * diff_anchor;
                    post_stds[i] += diff_compare * diff_compare;
                }
            }
            
            for (auto& std : post_stds) {
                std = std::sqrt(std / totalVectors);
            }
            
            std::cout << "\nPost-normalization statistics:"
                    << "\n  Mean range: [" << *std::min_element(post_means.begin(), post_means.end())
                    << ", " << *std::max_element(post_means.begin(), post_means.end()) << "]"
                    << "\n  Std range: [" << *std::min_element(post_stds.begin(), post_stds.end())
                    << ", " << *std::max_element(post_stds.begin(), post_stds.end()) << "]"
                    << "\n  Number of features: " << numFeatures << std::endl;
        }

        
        // Final shuffle of both sets
        std::shuffle(trainData.begin(), trainData.end(), gen);
        std::shuffle(valData.begin(), valData.end(), gen);
        
        // Print class distribution statistics
        size_t trainPos = 0, trainNeg = 0, valPos = 0, valNeg = 0;
        for (const auto& point : trainData) {
            if (point.label > 0.5f) trainPos++;
            else trainNeg++;
        }
        for (const auto& point : valData) {
            if (point.label > 0.5f) valPos++;
            else valNeg++;
        }
        
        std::cout << "\nFinal Split Statistics (after normalization):"
                << "\nTraining Set:"
                << "\n  Total: " << trainData.size()
                << "\n  Positive: " << trainPos << " (" 
                << (100.0f * trainPos / trainData.size()) << "%)"
                << "\n  Negative: " << trainNeg << " ("
                << (100.0f * trainNeg / trainData.size()) << "%)"
                << "\nValidation Set:"
                << "\n  Total: " << valData.size()
                << "\n  Positive: " << valPos << " ("
                << (100.0f * valPos / valData.size()) << "%)"
                << "\n  Negative: " << valNeg << " ("
                << (100.0f * valNeg / valData.size()) << "%)"
                << std::endl;
        
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

class AdaptiveMutation {
private:
    float base_rate;
    float min_rate;
    float max_rate;
    int stagnation_count;
    float last_best_fitness;

public:
    AdaptiveMutation(float base = 0.25f, float min = 0.1f, float max = 0.4f)
        : base_rate(base), min_rate(min), max_rate(max)
        , stagnation_count(0), last_best_fitness(std::numeric_limits<float>::max()) {}

    float getMutationRate(float current_best_fitness) {
        if (current_best_fitness >= last_best_fitness) {
            stagnation_count++;
            // Increase mutation rate during stagnation
            return std::min(max_rate, base_rate * (1.0f + 0.1f * stagnation_count));
        } else {
            stagnation_count = 0;
            // Decrease mutation rate when improving
            return std::max(min_rate, base_rate);
        }
        last_best_fitness = current_best_fitness;
    }
};

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
        
        // Validate inputs
        if (inputs.empty() || inputs[0].empty()) {
            return std::vector<float>{0.0f};
        }
        
        // Check for dropout
        if (shouldDropNode(node_id)) {
            return scaleForDropout(inputs[0]);
        }
        
        // Validate input sizes match
        size_t expected_size = inputs[0].size();
        for (const auto& input : inputs) {
            if (input.size() != expected_size) {
                return std::vector<float>{0.0f};
            }
        }
        
        // Perform operation
        std::vector<float> result;
        try {
            result = performOperation(op_name, inputs);
        } catch (const std::exception& e) {
            std::cerr << "Operation failed: " << e.what() << std::endl;
            return std::vector<float>{0.0f};
        }
        
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

class GPNode {
public:
    virtual ~GPNode() = default;
    virtual std::vector<float> evaluate(const std::vector<float>& input) = 0;
    virtual std::unique_ptr<GPNode> clone() const = 0;
    virtual int size() const = 0;
    virtual int depth() const = 0;
    virtual void mutate(std::mt19937& gen, const GPConfig& config) = 0;
    virtual std::vector<GPNode*> getAllNodes() = 0;  // Original
    virtual void replaceSubtree(GPNode* old_subtree, std::unique_ptr<GPNode> new_subtree) = 0;
    virtual bool isValidDepth() const {
        int d = depth();
        return d > 0 && d < 1000; // Reasonable upper limit
    }
    
    virtual int safeDepth() const {
        try {
            int d = depth();
            if (d < 0 || d > 1000) { // Reasonable limits
                std::cerr << "Warning: Invalid depth " << d << " detected" << std::endl;
                return 0;
            }
            return d;
        } catch (const std::exception& e) {
            std::cerr << "Error calculating depth: " << e.what() << std::endl;
            return 0;
        }
    }

    virtual bool isLeaf() const = 0;
    
    virtual bool isOperator() const = 0;
};

class FeatureNode : public GPNode {
private:
    int feature_index;
    static constexpr float SCALE_FACTOR = 0.01f;  // Scale factor to bring values to [-1, 1] range

public:
    explicit FeatureNode(int idx) : feature_index(idx) {}

    int getFeatureIndex() const { return feature_index; }

    std::vector<float> evaluate(const std::vector<float>& input) override {
        if (feature_index >= input.size()) {
            std::cerr << "Warning: Feature index " << feature_index 
                      << " out of bounds for input size " << input.size() << std::endl;
            return std::vector<float>{0.0f};
        }
        
        return std::vector<float>{input[feature_index]};
    }

    std::vector<GPNode*> getAllNodes() override {
        return {this};
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

    void replaceSubtree(GPNode* old_subtree, std::unique_ptr<GPNode> new_subtree) override {
        // No-op for leaf nodes unless it is the target node
        if (old_subtree == this) {
            throw std::runtime_error("Cannot replace root node in FeatureNode");
        }
    }

    bool isLeaf() const override { return true; }
    bool isOperator() const override { return false; }
};

class ConstantNode : public GPNode {
private:
    float value;

public:
    explicit ConstantNode(float v) : value(std::tanh(v)) {}  // Ensure value is in [-1, 1]

    float getValue() const { return value; }

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

    std::vector<GPNode*> getAllNodes() override {
        return {this};
    }

    void replaceSubtree(GPNode* old_subtree, std::unique_ptr<GPNode> new_subtree) override {
        // No-op for leaf nodes unless it is the target node
        if (old_subtree == this) {
            throw std::runtime_error("Cannot replace root node in ConstantNode");
        }
    }

    bool isLeaf() const override { return true; }
    bool isOperator() const override { return false; }
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

    void setOperator(const std::string& new_op) {
        // Validate operator arity matches current children
        bool is_new_unary = (new_op == "sin" || new_op == "cos" || new_op == "neg");
        bool is_current_unary = (children.size() == 1);
        
        // Only allow operator change if arity matches
        if (is_new_unary == is_current_unary) {
            op_name = new_op;
        } else {
            throw std::runtime_error("Cannot change operator type: arity mismatch");
        }
    }

    std::vector<float> evaluate(const std::vector<float>& input) override {
        if (children.empty()) {
            return std::vector<float>{0.0f};
        }
        
        std::vector<std::vector<float>> child_results;
        child_results.reserve(children.size());
        
        // Evaluate children and validate results
        bool valid_results = true;
        for (const auto& child : children) {
            if (!child) {
                valid_results = false;
                break;
            }
            
            auto result = child->evaluate(input);
            if (result.empty()) {
                valid_results = false;
                break;
            }
            
            child_results.push_back(std::move(result));
        }
        
        // Return safe default if any evaluation failed
        if (!valid_results) {
            return std::vector<float>{0.0f};
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

    void mutate(std::mt19937& gen, const GPConfig& config) override {
        if (children.empty()) return;
        
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        // Potentially mutate operator type
        if (prob_dist(gen) < 0.2f) {
            try {
                const bool is_unary = children.size() == 1;
                const std::vector<std::string> unary_ops = {"sin", "cos", "neg"};
                const std::vector<std::string> binary_ops = {"add", "sub", "mul", "div"};
                
                const auto& ops = is_unary ? unary_ops : binary_ops;
                std::uniform_int_distribution<size_t> op_dist(0, ops.size() - 1);
                setOperator(ops[op_dist(gen)]);
            } catch (const std::exception& e) {
                std::cerr << "Operator mutation failed: " << e.what() << std::endl;
            }
        }
        
        // Mutate a random child
        std::uniform_int_distribution<size_t> dist(0, children.size() - 1);
        size_t child_idx = dist(gen);
        
        if (children[child_idx]) {
            children[child_idx]->mutate(gen, config);
        }
    }

    std::vector<GPNode*> getAllNodes() override {
        std::vector<GPNode*> nodes = {this};
        for (const auto& child : children) {
            if (child) {
                auto child_nodes = child->getAllNodes();
                nodes.insert(nodes.end(), child_nodes.begin(), child_nodes.end());
            }
        }
        return nodes;
    }

    void replaceSubtree(GPNode* old_subtree, std::unique_ptr<GPNode> new_subtree) override {
        // Check if any direct child is the target
        for (auto& child : children) {
            if (child.get() == old_subtree) {
                child = std::move(new_subtree);
                return;
            }
        }

        // If not found in direct children, recurse into children
        for (auto& child : children) {
            if (child) {
                child->replaceSubtree(old_subtree, std::move(new_subtree));
            }
        }
    }

    int depth() const override {
        if (children.empty()) return 1;
        
        int max_child_depth = 0;
        for (const auto& child : children) {
            if (child) {
                // Add overflow protection
                try {
                    int child_depth = child->depth();
                    if (child_depth > 0 && child_depth < 1000) { // Validate child depth
                        max_child_depth = std::max(max_child_depth, child_depth);
                    } else {
                        std::cerr << "Warning: Invalid child depth " << child_depth << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error in depth calculation: " << e.what() << std::endl;
                }
            }
        }
        
        // Check for overflow before adding
        if (max_child_depth > INT_MAX - 1) {
            std::cerr << "Depth overflow detected" << std::endl;
            return INT_MAX;
        }
        
        return max_child_depth + 1;
    }
    
    // Add depth logging
    void logTreeDepth() const {
        std::cout << "Tree depth: " << depth() << std::endl;
        std::cout << "Number of children: " << children.size() << std::endl;
        for (size_t i = 0; i < children.size(); ++i) {
            if (children[i]) {
                std::cout << "Child " << i << " depth: " << children[i]->depth() << std::endl;
            }
        }
    }

    bool isLeaf() const override { return false; }
    bool isOperator() const override { return true; }
    const std::string& getOperatorName() const { return op_name; }
    const std::vector<std::unique_ptr<GPNode>>& getChildren() const {
        return children;
    }

    std::vector<std::unique_ptr<GPNode>>& getMutableChildren() {
        return children;
    }
};

class TreeOperations {
private:
    static thread_local int operation_counter;
    static constexpr int MAX_OPERATIONS = 5000;  // Increased from 1000
    static constexpr int MAX_RETRIES = 3;
    static constexpr float OPERATOR_PROB = 0.7f;
    static constexpr float FEATURE_NODE_PROB = 0.8f;

    // Reset operation counter
    static void resetCounter() {
        operation_counter = 0;
    }

    // Check if we've exceeded operation limit
    static bool checkOperationLimit() {
        if (++operation_counter > MAX_OPERATIONS) {
            std::cerr << "Warning: Operation limit exceeded (" << operation_counter << " operations)" << std::endl;
            resetCounter();
            return false;
        }
        return true;
    }

    // Helper function to create a leaf node
    static std::unique_ptr<GPNode> createLeafNode(const GPConfig& config, std::mt19937& gen) {
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        if (prob_dist(gen) < FEATURE_NODE_PROB) {
            std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
            return std::make_unique<FeatureNode>(feature_dist(gen));
        } else {
            std::uniform_real_distribution<float> const_dist(-1.0f, 1.0f);
            return std::make_unique<ConstantNode>(const_dist(gen));
        }
    }

public:

    static std::unique_ptr<GPNode> createRandomTree(
        int max_depth,
        int current_depth,
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        // At max_depth-1, only create leaf nodes to ensure we don't exceed max_depth
        if (current_depth >= max_depth - 1) {
            return createLeafNode(config, gen);
        }

        // If we haven't reached minimum depth, must create operator
        if (current_depth < config.min_tree_depth) {
            return createRandomOperatorNode(max_depth, current_depth, config, ops, gen);
        }

        // Between min and max depth, randomly choose
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        if (prob_dist(gen) < OPERATOR_PROB && current_depth < max_depth - 1) {
            return createRandomOperatorNode(max_depth, current_depth, config, ops, gen);
        }

        return createLeafNode(config, gen);
    }

    static std::unique_ptr<GPNode> createRandomOperatorNode(
        int max_depth,
        int current_depth,
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        // Ensure we have enough depth left to meet minimum requirements
        int remaining_depth = max_depth - current_depth;
        if (remaining_depth < 2) {  // Need at least 2 levels for operator + children
            return createFallbackTree(config, ops, gen);
        }

        static thread_local std::uniform_int_distribution<int> op_dist(0, 6);
        static const std::vector<std::pair<std::string, bool>> operators = {
            {"add", false}, {"sub", false}, {"mul", false}, 
            {"sin", true}, {"cos", true}, {"neg", true}, {"div", false}
        };

        // If we need to force more depth to meet minimum requirements
        bool need_more_depth = current_depth + 2 < config.min_tree_depth;

        const auto& [op_name, is_unary] = operators[op_dist(gen)];
        std::vector<std::unique_ptr<GPNode>> children;

        // For first child, recursively create a subtree
        if (need_more_depth) {
            // Force operator node creation to build depth
            auto first_child = createRandomOperatorNode(max_depth - 1, current_depth + 1, config, ops, gen);
            if (!first_child) return nullptr;
            children.push_back(std::move(first_child));
        } else {
            // Can use either operator or leaf node
            static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
            if (prob_dist(gen) < 0.7f) {  // 70% chance of operator for better depth
                auto first_child = createRandomOperatorNode(max_depth - 1, current_depth + 1, config, ops, gen);
                if (!first_child) return nullptr;
                children.push_back(std::move(first_child));
            } else {
                children.push_back(createLeafNode(config, gen));
            }
        }

        // For binary operators, create second child
        if (!is_unary) {
            if (need_more_depth) {
                // Force operator node creation for second child too
                auto second_child = createRandomOperatorNode(max_depth - 1, current_depth + 1, config, ops, gen);
                if (!second_child) return nullptr;
                children.push_back(std::move(second_child));
            } else {
                static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
                if (prob_dist(gen) < 0.7f) {
                    auto second_child = createRandomOperatorNode(max_depth - 1, current_depth + 1, config, ops, gen);
                    if (!second_child) return nullptr;
                    children.push_back(std::move(second_child));
                } else {
                    children.push_back(createLeafNode(config, gen));
                }
            }
        }

        auto node = std::make_unique<OperatorNode>(op_name, std::move(children), ops);
        
        // Verify the depth
        int actual_depth = node->depth();
        if (actual_depth < config.min_tree_depth || actual_depth >= config.max_tree_depth) {
            return createFallbackTree(config, ops, gen);
        }
        
        return node;
    }

    
    static std::unique_ptr<GPNode> multiPointCrossover(
        const GPNode* parent1,
        const GPNode* parent2,
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        if (!parent1 || !parent2) return nullptr;
        
        // Create initial copy of parent1
        auto result = parent1->clone();
        if (!result || result->depth() >= config.max_tree_depth) {
            return createFallbackTree(config, ops, gen);
        }

        auto temp_parent2 = parent2->clone();
        if (!temp_parent2) return result;

        std::vector<GPNode*> nodes1 = result->getAllNodes();
        std::vector<GPNode*> nodes2 = temp_parent2->getAllNodes();
        
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        int attempts = 0;
        const int MAX_ATTEMPTS = 5;
        
        while (attempts < MAX_ATTEMPTS) {
            // Select random nodes
            std::uniform_int_distribution<size_t> idx_dist1(0, nodes1.size() - 1);
            std::uniform_int_distribution<size_t> idx_dist2(0, nodes2.size() - 1);
            
            size_t idx1 = idx_dist1(gen);
            size_t idx2 = idx_dist2(gen);
            
            // Create a copy to test the crossover
            auto test_tree = result->clone();
            if (!test_tree) continue;
            
            // Clone the subtree we want to insert
            auto new_subtree = nodes2[idx2]->clone();
            if (!new_subtree) continue;
            
            try {
                // Try the replacement on the test tree
                test_tree->replaceSubtree(nodes1[idx1], std::move(new_subtree));
                
                // Verify depth constraints
                int new_depth = test_tree->depth();
                if (new_depth >= config.min_tree_depth && new_depth < config.max_tree_depth) {
                    // If valid, perform the actual replacement
                    auto actual_subtree = nodes2[idx2]->clone();
                    result->replaceSubtree(nodes1[idx1], std::move(actual_subtree));
                    return result;
                }
            } catch (const std::exception&) {
                // If replacement fails, try again
            }
            
            attempts++;
        }
        
        // If all attempts fail, return fallback tree
        return createFallbackTree(config, ops, gen);
    }

    static void mutateNode(
        GPNode* node,
        const GPConfig& config,
        std::mt19937& gen,
        int current_depth = 0,
        int max_allowed_depth = 0  // New parameter to track allowed depth
    ) {
        if (!node) return;
        
        if (max_allowed_depth == 0) {
            // Initialize max_allowed_depth on first call
            max_allowed_depth = config.max_tree_depth;
        }
        
        // Stop if we've reached the maximum allowed depth
        if (current_depth >= max_allowed_depth) return;
        
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        if (node->isOperator()) {
            auto* op_node = dynamic_cast<OperatorNode*>(node);
            if (!op_node) return;

            // Less aggressive operator mutation
            if (prob_dist(gen) < 0.1f) {
                const bool is_unary = op_node->getChildren().size() == 1;
                const std::vector<std::string>& ops = is_unary ? 
                    std::vector<std::string>{"sin", "cos", "neg"} :
                    std::vector<std::string>{"add", "sub", "mul", "div"};
                
                std::uniform_int_distribution<size_t> op_dist(0, ops.size() - 1);
                try {
                    op_node->setOperator(ops[op_dist(gen)]);
                } catch (const std::exception&) {
                    // Ignore operator mutation failures
                }
            }
            
            // Available depth for children
            int remaining_depth = max_allowed_depth - current_depth - 1;
            if (remaining_depth <= 0) return;
            
            // More conservative child mutation probability
            float depth_factor = std::max(0.0f, 
                1.0f - (static_cast<float>(current_depth) / (config.max_tree_depth - 1)));
            float mut_prob = config.mutation_prob * depth_factor * 0.5f;
            
            auto& children = op_node->getMutableChildren();
            for (auto& child : children) {
                if (child && prob_dist(gen) < mut_prob) {
                    // Create backup
                    auto backup = child->clone();
                    auto* child_ptr = child.get();
                    
                    // Attempt mutation with depth limit
                    mutateNode(child_ptr, config, gen, current_depth + 1, remaining_depth);
                    
                    // Verify depth after mutation
                    if (child_ptr->depth() > remaining_depth) {
                        child = std::move(backup);
                    }
                }
            }
        } else if (prob_dist(gen) < config.mutation_prob) {
            // For leaf nodes, just mutate normally as they can't increase depth
            try {
                node->mutate(gen, config);
            } catch (const std::exception&) {
                // Ignore leaf mutation failures
            }
        }
    }

    static std::unique_ptr<GPNode> createFallbackTree(
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        // Create a tree with exactly minimum depth (3)
        std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
        
        // Create level 3 (leaf nodes)
        auto leaf1 = std::make_unique<FeatureNode>(feature_dist(gen));
        auto leaf2 = std::make_unique<FeatureNode>(feature_dist(gen));
        auto leaf3 = std::make_unique<FeatureNode>(feature_dist(gen));
        auto leaf4 = std::make_unique<FeatureNode>(feature_dist(gen));
        
        // Create level 2 (inner nodes)
        std::vector<std::unique_ptr<GPNode>> inner1_children;
        inner1_children.push_back(std::move(leaf1));
        inner1_children.push_back(std::move(leaf2));
        auto inner1 = std::make_unique<OperatorNode>("add", std::move(inner1_children), ops);
        
        std::vector<std::unique_ptr<GPNode>> inner2_children;
        inner2_children.push_back(std::move(leaf3));
        inner2_children.push_back(std::move(leaf4));
        auto inner2 = std::make_unique<OperatorNode>("mul", std::move(inner2_children), ops);
        
        // Create root (level 1)
        std::vector<std::unique_ptr<GPNode>> root_children;
        root_children.push_back(std::move(inner1));
        root_children.push_back(std::move(inner2));
        
        return std::make_unique<OperatorNode>("add", std::move(root_children), ops);
    }
};

// Initialize the thread_local operation counter
thread_local int TreeOperations::operation_counter = 0;

// Individual class representing a collection of trees
class Individual {
private:
    std::vector<std::unique_ptr<GPNode>> trees;
    float fitness;
    std::shared_ptr<GPOperations> ops;
    const GPConfig& config;  // Reference to config
    mutable std::mutex eval_mutex;
    mutable std::atomic<int> eval_count{0};
    mutable std::atomic<bool> evaluation_in_progress{false};

    size_t computeTreeHash(const std::unique_ptr<GPNode>& tree) const {
        if (!tree) return 0;
        
        // Use node visitor pattern to build hash
        std::vector<GPNode*> nodes = tree->getAllNodes();
        size_t hash = 0;
        
        for (const GPNode* node : nodes) {
            // Hash node type and data
            if (const auto* op_node = dynamic_cast<const OperatorNode*>(node)) {
                // Hash operator nodes - combine operator name and arity
                hash = hashCombine(hash, std::hash<std::string>{}(op_node->getOperatorName()));
                hash = hashCombine(hash, op_node->getChildren().size());
            }
            else if (const auto* feature_node = dynamic_cast<const FeatureNode*>(node)) {
                // Hash feature nodes - include feature index
                hash = hashCombine(hash, 0x1234567); // Magic number for feature node type
                hash = hashCombine(hash, feature_node->getFeatureIndex());
            }
            else if (const auto* constant_node = dynamic_cast<const ConstantNode*>(node)) {
                // Hash constant nodes - include value
                hash = hashCombine(hash, 0x89ABCDEF); // Magic number for constant node type
                hash = hashCombine(hash, constant_node->getValue());
            }
        }
        return hash;
    }
    
    // Helper function to combine hash values
    static size_t hashCombine(size_t seed, size_t value) {
        seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

public:
    Individual(
        std::vector<std::unique_ptr<GPNode>> t, 
        std::shared_ptr<GPOperations> operations,
        const GPConfig& cfg
    ) : trees(std::move(t))
      , fitness(std::numeric_limits<float>::infinity())
      , ops(operations)
      , config(cfg)
    {
        if (trees.empty()) {
            throw std::runtime_error("Cannot create Individual with empty tree vector");
        }
        if (!ops) {
            throw std::runtime_error("Cannot create Individual with null operations");
        }
    }

    // Deep copy constructor - properly initialize config reference
    Individual(const Individual& other) 
        : fitness(other.fitness)
        , ops(other.ops)
        , config(other.config)  // Initialize reference from other
    {
        std::lock_guard<std::mutex> lock(other.eval_mutex);
        trees.reserve(other.trees.size());
        for (const auto& tree : other.trees) {
            trees.push_back(tree->clone());
        }
    }

    // Move constructor - properly initialize config reference
    Individual(Individual&& other) noexcept
        : trees(std::move(other.trees))
        , fitness(other.fitness)
        , ops(std::move(other.ops))
        , config(other.config)  // Initialize reference from other
    {}

    // Copy assignment operator
    Individual& operator=(Individual other) {
        std::swap(trees, other.trees);
        fitness = other.fitness;
        ops = other.ops;
        // Note: We can't swap config as it's a reference
        return *this;
    }

    std::vector<float> evaluate(const std::vector<float>& input) const {
        std::vector<float> result(1, 0.0f);
        std::lock_guard<std::mutex> lock(eval_mutex);
        
        try {
            if (input.empty() || trees.empty()) {
                return result;
            }

            size_t valid_evaluations = 0;
            const size_t max_retries = 3;
            
            for (size_t i = 0; i < trees.size(); ++i) {
                size_t retries = 0;
                bool success = false;
                
                while (retries < max_retries && !success) {
                    try {
                        // Validate tree
                        if (!trees[i]) {
                            break;
                        }
                        
                        // Check depth before evaluation
                        int depth = trees[i]->depth();
                        if (depth < 0 || depth >= config.max_tree_depth) {
                            break;
                        }

                        // Evaluate with state tracking
                        auto tree_result = trees[i]->evaluate(input);
                        
                        // Validate result immediately
                        if (tree_result.empty()) {
                            retries++;
                            continue;
                        }

                        // Update result safely
                        result[0] += tree_result[0];
                        valid_evaluations++;
                        success = true;
                    }
                    catch (...) {
                        retries++;
                    }
                }
            }

            // Average only if we have valid results
            if (valid_evaluations > 0) {
                result[0] /= static_cast<float>(valid_evaluations);
            }
        }
        catch (...) {
            // Return default on catastrophic failure
            return std::vector<float>(1, 0.0f);
        }

        return result;
    }

    void setFitness(float f) { fitness = f; }
    float getFitness() const { return fitness; }

    void mutate(std::mt19937& gen) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& tree : trees) {
            if (dist(gen) < config.mutation_prob) {
                if (tree && tree->isValidDepth()) {
                    TreeOperations::mutateNode(tree.get(), config, gen);
                } else {
                    // If tree is invalid, create a new valid tree
                    tree = TreeOperations::createRandomOperatorNode(
                        config.max_tree_depth, 0, config, ops, gen);
                }
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

    size_t computeHash() const {
        size_t hash = 0;
        
        // Hash each tree in the individual
        for (const auto& tree : trees) {
            hash = hashCombine(hash, computeTreeHash(tree));
        }
        
        // Include fitness in hash to differentiate between similar but not identical individuals
        hash = hashCombine(hash, std::hash<float>{}(fitness));
        
        return hash;
    }

    void debugPrintTreeDepths() const {
        std::cout << "Individual tree depths:" << std::endl;
        for (size_t i = 0; i < trees.size(); ++i) {
            if (trees[i]) {
                int depth = trees[i]->depth();
                std::cout << "  Tree " << i << ": depth = " << depth 
                         << " (valid: " << (depth >= config.min_tree_depth && 
                                          depth <= config.max_tree_depth) << ")" << std::endl;
            } else {
                std::cout << "  Tree " << i << ": NULL" << std::endl;
            }
        }
    }

    bool validateDepth() const {
        bool valid = true;
        for (const auto& tree : trees) {
            if (!tree) {
                std::cout << "Found null tree in individual" << std::endl;
                valid = false;
                continue;
            }
            
            int depth = tree->depth();
            if (depth < config.min_tree_depth || depth > config.max_tree_depth) {
                std::cout << "Invalid tree depth: " << depth 
                         << " (min: " << config.min_tree_depth 
                         << ", max: " << config.max_tree_depth << ")" << std::endl;
                valid = false;
            }
        }
        return valid;
    }
};

namespace std {
    template<>
    struct hash<Individual> {
        size_t operator()(const Individual& ind) const {
            return ind.computeHash();
        }
    };

    // Also implement equality comparison for hash map lookups
    template<>
    struct equal_to<Individual> {
        bool operator()(const Individual& lhs, const Individual& rhs) const {
            // Compare fitness first as it's cheaper
            if (std::abs(lhs.getFitness() - rhs.getFitness()) > 1e-6f) {
                return false;
            }
            
            // Then compare tree structures
            return lhs.computeHash() == rhs.computeHash();
        }
    };
}

class FitnessCache {
private:
    std::unordered_map<Individual, float> cache;
    std::mutex cache_mutex;
    static constexpr size_t MAX_CACHE_SIZE = 10000;

public:
    void clear() {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.clear();
    }

public:
    FitnessCache() = default;
    FitnessCache(const FitnessCache&) = delete;
    FitnessCache& operator=(const FitnessCache&) = delete;

    std::optional<float> get(const Individual& ind) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(ind);
        if (it != cache.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    void put(const Individual& ind, float fitness) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (cache.size() >= MAX_CACHE_SIZE) {
            // Simple eviction strategy: clear half the cache
            auto it = cache.begin();
            for (size_t i = 0; i < MAX_CACHE_SIZE / 2 && it != cache.end(); ++i) {
                it = cache.erase(it);
            }
        }
        cache.emplace(ind, fitness);
    }
};


// Profiling class to track timings
// Updated Profiler class
class Profiler {
private:
    struct TimingData {
        double total_time = 0.0;
        size_t calls = 0;
    };
    
    std::map<std::string, TimingData> timings;
    std::chrono::high_resolution_clock::time_point generation_start;
    
public:
    void startGeneration() {
        generation_start = std::chrono::high_resolution_clock::now();
    }
    
    double getGenerationTime() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - generation_start).count();
    }
    
    void recordTime(const std::string& operation, double seconds) {
        timings[operation].total_time += seconds;
        timings[operation].calls++;
    }
    
    // Add this method to expose timings
    const std::map<std::string, TimingData>& getTimings() const {
        return timings;
    }
    
    void printStatistics(int generation) {
        std::cout << "\nProfiling Statistics for Generation " << generation << ":\n";
        for (const auto& [operation, data] : timings) {
            double avg_time = data.total_time / data.calls;
            std::cout << "  " << operation << ":\n"
                     << "    Total time: " << data.total_time << "s\n"
                     << "    Calls: " << data.calls << "\n"
                     << "    Average time: " << avg_time << "s\n";
        }
        std::cout << std::endl;
        
        // Clear timings for next generation
        timings.clear();
    }
};


class ContrastiveGP {
private:
    GPConfig config;
    std::shared_ptr<GPOperations> ops;
    std::vector<Individual> population;
    static thread_local std::mt19937 rng;
    FitnessCache fitness_cache; // Add as class member

    std::mt19937& getGen() {
        return rng;  // Now returns reference to thread-local RNG
    }

    Individual createRandomIndividual() {
        std::vector<std::unique_ptr<GPNode>> trees;
        trees.reserve(config.num_trees);
        
        for (int i = 0; i < config.num_trees; ++i) {
            auto tree = TreeOperations::createRandomOperatorNode(
                config.max_tree_depth, 0, config, ops, getGen());
            if (!tree) {
                // If tree creation fails, try again with reduced depth
                tree = TreeOperations::createRandomOperatorNode(
                    config.max_tree_depth - 1, 0, config, ops, getGen());
            }
            if (!tree) {
                // If still fails, create minimal valid tree
                tree = TreeOperations::createRandomOperatorNode(
                    config.min_tree_depth + 1, 0, config, ops, getGen());
            }
            trees.push_back(std::move(tree));
        }
        
        return Individual(std::move(trees), ops, config);
    }

    float calculateDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
        if (v1.empty() || v2.empty() || v1.size() != v2.size()) {
            std::cerr << "Warning: Empty or mismatched vectors in distance calculation. "
                    << "Sizes: " << v1.size() << " and " << v2.size() << std::endl;
            return std::numeric_limits<float>::max();
        }

        // Calculate dot product and magnitudes
        float dot_product = 0.0f;
        float mag1 = 0.0f;
        float mag2 = 0.0f;
        
        for (size_t i = 0; i < v1.size(); ++i) {
            dot_product += v1[i] * v2[i];
            mag1 += v1[i] * v1[i];
            mag2 += v2[i] * v2[i];
        }

        // Add small epsilon to prevent division by zero
        constexpr float epsilon = 1e-8f;
        mag1 = std::sqrt(mag1 + epsilon);
        mag2 = std::sqrt(mag2 + epsilon);

        // Calculate cosine similarity
        float similarity = dot_product / (mag1 * mag2);
        
        // Clamp similarity to [-1, 1] range to handle numerical instability
        similarity = std::max(-1.0f, std::min(1.0f, similarity));
        
        // Convert similarity to distance (1 - similarity) and scale to [0, 2] range
        float distance = 1.0f - similarity;
        
        // Check for NaN or inf
        if (std::isnan(distance) || std::isinf(distance)) {
            std::cerr << "Warning: Invalid distance calculated" << std::endl;
            return std::numeric_limits<float>::max();
        }
        
        return distance;
    }

    float evaluateIndividual(const Individual& ind, const std::vector<DataPoint>& data) {
        // Try to get cached fitness first
        if (auto cached_fitness = fitness_cache.get(ind)) {
            return *cached_fitness;
        }

        if (data.empty()) {
            std::cerr << "Warning: Empty dataset in evaluation" << std::endl;
            return std::numeric_limits<float>::max();
        }

        float total_loss = 0.0f;
        int valid_pairs = 0;
        int failed_evaluations = 0;
        constexpr int MAX_FAILURES = 10;

        for (size_t i = 0; i < data.size() && failed_evaluations < MAX_FAILURES; i += config.batch_size) {
            size_t batch_end = std::min(i + config.batch_size, data.size());
            
            for (size_t j = i; j < batch_end; ++j) {
                const auto& point = data[j];
                try {
                    if (point.anchor.empty() || point.compare.empty()) {
                        std::cerr << "Empty vectors in data point " << j << std::endl;
                        continue;
                    }

                    if (point.anchor.size() != point.compare.size()) {
                        std::cerr << "Mismatched vector sizes in data point " << j << ": "
                                << point.anchor.size() << " vs " << point.compare.size() << std::endl;
                        continue;
                    }

                    auto anchor_output = ind.evaluate(point.anchor);
                    auto compare_output = ind.evaluate(point.compare);
                    
                    if (anchor_output.empty() || compare_output.empty()) {
                        std::cerr << "Empty output from individual evaluation at point " << j << std::endl;
                        failed_evaluations++;
                        continue;
                    }
                    
                    if (anchor_output.size() != compare_output.size()) {
                        std::cerr << "Mismatched output sizes at point " << j << ": "
                                << anchor_output.size() << " vs " << compare_output.size() << std::endl;
                        failed_evaluations++;
                        continue;
                    }
                    
                    float distance = calculateDistance(anchor_output, compare_output);
                    if (std::isnan(distance) || std::isinf(distance)) {
                        std::cerr << "Invalid distance calculated at point " << j << std::endl;
                        failed_evaluations++;
                        continue;
                    }
                    
                    float loss;
                    if (point.label > 0.5f) {
                        loss = distance * distance;
                    } else {
                        float margin_diff = std::max(0.0f, config.margin - distance);
                        loss = margin_diff * margin_diff;
                    }
                    
                    total_loss += loss;
                    valid_pairs++;
                    
                } catch (const std::exception& e) {
                    std::cerr << "Error evaluating point " << j << ": " << e.what() << std::endl;
                    failed_evaluations++;
                }
            }
        }
        
        if (failed_evaluations >= MAX_FAILURES) {
            std::cerr << "Too many evaluation failures" << std::endl;
            return std::numeric_limits<float>::max();
        }
        
        if (valid_pairs == 0) {
            std::cerr << "No valid evaluations completed" << std::endl;
            return std::numeric_limits<float>::max();
        }
        
        float avg_loss = total_loss / valid_pairs;
        float complexity_penalty = config.parsimony_coeff * ind.totalSize();
        float fitness = avg_loss + complexity_penalty;
        
        // Cache the result
        fitness_cache.put(ind, fitness);
        return fitness;
    }

    float calculateAccuracy(const Individual& ind, const std::vector<DataPoint>& data) {
        int true_positives = 0;
        int true_negatives = 0;
        int total_positives = 0;
        int total_negatives = 0;
        
        const size_t SAFE_BATCH_SIZE = 32;
        
        for (size_t i = 0; i < data.size(); i += SAFE_BATCH_SIZE) {
            size_t batch_end = std::min(i + SAFE_BATCH_SIZE, data.size());
            
            for (size_t j = i; j < batch_end; ++j) {
                const auto& point = data[j];
                
                try {
                    if (point.anchor.empty() || point.compare.empty()) {
                        continue;
                    }

                    auto anchor_output = ind.evaluate(point.anchor);
                    auto compare_output = ind.evaluate(point.compare);

                    if (anchor_output.empty() || compare_output.empty()) {
                        continue;
                    }

                    float distance = calculateDistance(anchor_output, compare_output);
                    bool prediction = distance < config.distance_threshold;
                    bool actual = point.label;

                    if (actual) {
                        total_positives++;
                        if (prediction) true_positives++;
                    } else {
                        total_negatives++;
                        if (!prediction) true_negatives++;
                    }

                } catch (const std::exception& e) {
                    std::cerr << "Evaluation failed: " << e.what() << std::endl;
                    continue;
                }
            }
        }

        if (total_positives == 0 || total_negatives == 0) {
            return 0.0f;
        }

        float sensitivity = static_cast<float>(true_positives) / total_positives;
        float specificity = static_cast<float>(true_negatives) / total_negatives;
        return (sensitivity + specificity) / 2.0f;
    }

    // Helper method to run a single tournament
    Individual& runTournament(const std::vector<size_t>& available_indices) {
        if (available_indices.empty()) {
            throw std::runtime_error("No individuals available for tournament");
        }

        // Create a temporary vector for tournament candidates
        std::vector<size_t> tournament_candidates;
        tournament_candidates.reserve(std::min((size_t)config.tournament_size, available_indices.size()));

        // Use local RNG
        auto& local_gen = getGen();

        // Fill tournament pool - ensure we don't exceed available indices
        for (int i = 0; i < config.tournament_size && i < available_indices.size(); ++i) {
            std::uniform_int_distribution<size_t> idx_dist(0, available_indices.size() - 1);
            tournament_candidates.push_back(available_indices[idx_dist(local_gen)]);
        }

        // Find the best individual in the tournament
        size_t best_idx = tournament_candidates[0];
        float best_fitness = std::numeric_limits<float>::max();  // Initialize to worst possible fitness

        // Safely find the best individual
        for (size_t idx : tournament_candidates) {
            if (idx < population.size()) {  // Bounds check
                float current_fitness = population[idx].getFitness();
                if (current_fitness < best_fitness) {
                    best_fitness = current_fitness;
                    best_idx = idx;
                }
            }
        }

        // Ensure best_idx is valid
        if (best_idx >= population.size()) {
            best_idx = 0;  // Fallback to first individual if something went wrong
        }

        return population[best_idx];
    }

    void evaluatePopulation(const std::vector<DataPoint>& trainData) {
        std::cout << "\nStarting population evaluation..." << std::endl;
        
        // Log initial state
        std::cout << "Population size: " << population.size() << std::endl;
        std::cout << "Training data size: " << trainData.size() << std::endl;
        
        static int generation_count = 0;
        generation_count++;
        
        std::cout << "Generation: " << generation_count << std::endl;

        // Add timing instrumentation
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Add depth monitoring with progress updates
        int max_depth_seen = 0;
        int total_depth = 0;
        int valid_trees = 0;
        
        std::cout << "Analyzing tree depths..." << std::endl;
        for (size_t i = 0; i < population.size(); ++i) {
            if (i % 10 == 0) {
                std::cout << "Processed " << i << "/" << population.size() << " individuals" << std::endl;
            }
            
            const auto& individual = population[i];
            for (int j = 0; j < config.num_trees; ++j) {
                const auto& tree = individual.getTree(j);
                if (tree) {
                    int tree_depth = tree->safeDepth();
                    max_depth_seen = std::max(max_depth_seen, tree_depth);
                    if (tree_depth > 0) {
                        total_depth += tree_depth;
                        valid_trees++;
                    }
                }
            }
        }
        
        // Log depth statistics
        std::cout << "Depth analysis complete:" << std::endl;
        if (valid_trees > 0) {
            float avg_depth = static_cast<float>(total_depth) / valid_trees;
            std::cout << "  Max depth: " << max_depth_seen << std::endl;
            std::cout << "  Average depth: " << avg_depth << std::endl;
            std::cout << "  Valid trees: " << valid_trees << "/" 
                    << (population.size() * config.num_trees) << std::endl;
        }

        const size_t num_workers = std::min((size_t)config.num_workers, population.size());
        const size_t batch_size = (population.size() + num_workers - 1) / num_workers;

        std::cout << "Starting parallel evaluation with:" << std::endl;
        std::cout << "  Number of workers: " << num_workers << std::endl;
        std::cout << "  Batch size: " << batch_size << std::endl;

        std::vector<std::future<void>> futures;
        std::atomic<int> completed_individuals{0};
        
        for (size_t i = 0; i < population.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, population.size());
            futures.push_back(std::async(std::launch::async, [this, i, end, &trainData, &completed_individuals]() {
                for (size_t j = i; j < end; ++j) {
                    try {
                        float fitness = evaluateIndividual(population[j], trainData);
                        population[j].setFitness(fitness);
                        int completed = ++completed_individuals;
                        if (completed % 10 == 0) {
                            std::cout << "Completed evaluating " << completed << " individuals" << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error evaluating individual " << j << ": " << e.what() << std::endl;
                    }
                }
            }));
        }
        
        std::cout << "Waiting for all evaluations to complete..." << std::endl;
        for (auto& future : futures) {
            future.wait();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "Population evaluation completed in " << duration.count() << " seconds" << std::endl;
    }
    
public:
    ContrastiveGP(const GPConfig& cfg) 
        : config(cfg)
        , ops(std::make_shared<GPOperations>(
            cfg.dropout_prob, cfg.bn_momentum, cfg.bn_epsilon))
    {
        population.reserve(config.population_size);
        for (int i = 0; i < config.population_size; ++i) {
            population.push_back(createRandomIndividual());
        }
    }


    // Method to select multiple parents using tournament selection
    std::vector<std::pair<Individual*, Individual*>> selectParents(size_t num_pairs_needed) {
        std::vector<std::pair<Individual*, Individual*>> selected_pairs;
        selected_pairs.reserve(num_pairs_needed);
        
        // Create indices for available individuals
        std::vector<size_t> available_indices(population.size());
        std::iota(available_indices.begin(), available_indices.end(), 0);

        // Create batches for parallel processing
        const size_t num_workers = std::min((size_t)config.num_workers, num_pairs_needed);
        const size_t pairs_per_worker = (num_pairs_needed + num_workers - 1) / num_workers;
        
        std::vector<std::future<std::vector<std::pair<Individual*, Individual*>>>> futures;
        std::mutex selection_mutex;  // For thread-safe access to shared resources

        // Launch parallel tournament selections
        for (size_t worker = 0; worker < num_workers; ++worker) {
            size_t start_pair = worker * pairs_per_worker;
            size_t end_pair = std::min(start_pair + pairs_per_worker, num_pairs_needed);
            
            if (start_pair >= end_pair) break;

            futures.push_back(std::async(std::launch::async, [this, start_pair, end_pair, &available_indices, &selection_mutex]() {
                std::vector<std::pair<Individual*, Individual*>> worker_pairs;
                worker_pairs.reserve(end_pair - start_pair);
                
                auto& local_gen = getGen();  // Thread-local RNG
                
                for (size_t i = start_pair; i < end_pair; ++i) {
                    std::lock_guard<std::mutex> lock(selection_mutex);
                    
                    // Run tournaments to select two parents
                    Individual& parent1 = runTournament(available_indices);
                    Individual& parent2 = runTournament(available_indices);
                    
                    // Add selected pair
                    worker_pairs.emplace_back(&parent1, &parent2);
                }
                
                return worker_pairs;
            }));
        }

        // Collect results from all workers
        for (auto& future : futures) {
            auto worker_results = future.get();
            selected_pairs.insert(selected_pairs.end(), 
                                worker_results.begin(), 
                                worker_results.end());
        }

        return selected_pairs;
    }

    void train(const std::vector<DataPoint>& trainData, const std::vector<DataPoint>& valData) {
        Profiler profiler;
        float best_fitness = std::numeric_limits<float>::max();
        int generations_without_improvement = 0;
        const size_t EVAL_SUBSET_SIZE = 100;  // Size of random subset for evaluation

        // RNG for selecting random training samples
        std::random_device rd;
        std::mt19937 subset_gen(rd());

        for (int generation = 0; generation < config.generations; ++generation) {
            profiler.startGeneration();
            std::cout << "\nStarting generation " << generation << std::endl;
            
            // Create random subset of training data for this generation
            std::vector<DataPoint> train_subset;
            train_subset.reserve(EVAL_SUBSET_SIZE);
            
            if (trainData.size() <= EVAL_SUBSET_SIZE) {
                train_subset = trainData;  // Use all data if we have less than subset size
            } else {
                // Create indices for sampling
                std::vector<size_t> indices(trainData.size());
                std::iota(indices.begin(), indices.end(), 0);
                
                // Randomly shuffle and take first EVAL_SUBSET_SIZE elements
                std::shuffle(indices.begin(), indices.end(), subset_gen);
                for (size_t i = 0; i < EVAL_SUBSET_SIZE; ++i) {
                    train_subset.push_back(trainData[indices[i]]);
                }
            }

            std::cout << "Evaluate population on susbet" << std::endl;
            
            // Evaluate current population using the subset
            evaluatePopulation(train_subset);
            
            // Sort population by fitness
            std::vector<size_t> indices(population.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                    [this](size_t a, size_t b) { 
                        return population[a].getFitness() < population[b].getFitness(); 
                    });

            // Create new population vector
            std::vector<Individual> new_population;
            new_population.reserve(config.population_size);

            std::cout << "Elitism" << std::endl;

            // Add elite individuals first
            for (int i = 0; i < config.elite_size && i < indices.size(); ++i) {
                new_population.push_back(Individual(population[indices[i]]));
            }

            // Keep track of failed attempts to prevent infinite loops
            std::cout << "Starting crossover and mutation..." << std::endl;

            while (new_population.size() < config.population_size) {
                std::cout << "Starting new iteration, current population size: " 
                        << new_population.size() << std::endl;
                
                // Select parents using tournament selection
                Individual& parent1 = runTournament(indices);
                Individual& parent2 = runTournament(indices);
                
                std::cout << "Parents selected. Validating..." << std::endl;
                // Validate parents
                if (!parent1.validateDepth() || !parent2.validateDepth()) {
                    std::cout << "Invalid parent depth detected, creating random individual" << std::endl;
                    new_population.push_back(createRandomIndividual());
                    continue;
                }
                
                std::cout << "Creating offspring through crossover..." << std::endl;
                // Create offspring through crossover
                Individual offspring(crossover(parent1, parent2));
                
                std::cout << "Validating offspring..." << std::endl;
                // Validate offspring
                if (!offspring.validateDepth()) {
                    std::cout << "Invalid offspring depth detected, creating random individual" << std::endl;
                    new_population.push_back(createRandomIndividual());
                    continue;
                }
                
                std::cout << "Applying mutation..." << std::endl;
                // Apply mutation
                if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < config.mutation_prob) {
                    offspring.mutate(rng);
                    // Validate after mutation
                    if (!offspring.validateDepth()) {
                        std::cout << "Invalid depth after mutation, creating random individual" << std::endl;
                        new_population.push_back(createRandomIndividual());
                        continue;
                    }
                }

                std::cout << "Adding offspring to population..." << std::endl;
                // Verify offspring is still valid
                if (!offspring.validateDepth()) {
                    std::cerr << "Invalid offspring before adding to population!" << std::endl;
                    return;
                }
                new_population.push_back(std::move(offspring));
                std::cout << "Successfully added offspring" << std::endl;
            }

            // Fill any remaining slots with random individuals
            while (new_population.size() < config.population_size) {
                new_population.push_back(createRandomIndividual());
            }

            // Calculate statistics for current generation
            float gen_best_fitness = std::numeric_limits<float>::max();
            float gen_avg_fitness = 0.0f;
            
            for (const auto& individual : new_population) {
                float fitness = individual.getFitness();
                gen_avg_fitness += fitness;
                gen_best_fitness = std::min(gen_best_fitness, fitness);
            }
            gen_avg_fitness /= new_population.size();

            std::cout << "Calculating accuracy" << std::endl;

            // Calculate accuracies on the full training and validation sets
            // for the best individual to get true performance metrics
            std::cout << "Best: " << new_population[0].getFitness() << std::endl;
        
            float train_accuracy = calculateAccuracy(new_population[0], trainData);
            ops->setTraining(false);
            float val_accuracy = calculateAccuracy(new_population[0], valData);
            ops->setTraining(true);

            // Output statistics
            std::cout << "Generation " << generation 
                    << "\n  Best Fitness (on subset): " << gen_best_fitness
                    << "\n  Avg Fitness (on subset): " << gen_avg_fitness
                    << "\n  Training Accuracy (full set): " << train_accuracy * 100.0f << "%"
                    << "\n  Validation Accuracy: " << val_accuracy * 100.0f << "%"
                    << "\n  Population Size: " << new_population.size()
                    << "\n  Training Subset Size: " << train_subset.size()
                    << std::endl;

            std::cout << "Updating population" << std::endl;

            // Update population
            population = std::move(new_population);

            // Update best fitness and check for improvement
            if (gen_best_fitness < best_fitness) {
                best_fitness = gen_best_fitness;
                generations_without_improvement = 0;
            } else {
                generations_without_improvement++;
            }

            // Every 10 generations, clear the fitness cache to prevent memory bloat
            if (generation % 10 == 0) {
                fitness_cache.clear();
            }
        }
    }
    
    Individual crossover(const Individual& parent1, const Individual& parent2) {
        std::vector<std::unique_ptr<GPNode>> offspring_trees;
        offspring_trees.reserve(config.num_trees);
        
        std::cout << "Starting crossover process..." << std::endl;
        
        // Track mutations and crossovers performed to encourage diversity
        int mutations_performed = 0;
        int crossovers_performed = 0;
        
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        for (int i = 0; i < config.num_trees; ++i) {
            auto parent1_tree = parent1.getTree(i).get();
            auto parent2_tree = parent2.getTree(i).get();
            
            bool should_crossover = prob_dist(getGen()) < config.crossover_prob;
            
            if (should_crossover && parent1_tree && parent2_tree && 
                parent1_tree->isValidDepth() && parent2_tree->isValidDepth()) {
                try {
                    auto new_tree = TreeOperations::multiPointCrossover(
                        parent1_tree, parent2_tree, config, ops, getGen());
                    
                    if (new_tree && new_tree->depth() >= config.min_tree_depth && 
                        new_tree->depth() < config.max_tree_depth) {
                        offspring_trees.push_back(std::move(new_tree));
                        crossovers_performed++;
                        continue;
                    }
                } catch (const std::exception& e) {
                    std::cout << "Crossover failed for tree " << i << ": " << e.what() << std::endl;
                }
            }
            
            // If crossover fails or isn't performed, try mutation of parent1's tree
            try {
                auto new_tree = parent1.getTree(i)->clone();
                if (new_tree && new_tree->isValidDepth()) {
                    TreeOperations::mutateNode(new_tree.get(), config, getGen(), 0, config.max_tree_depth - 1);
                    
                    if (new_tree->depth() >= config.min_tree_depth && 
                        new_tree->depth() < config.max_tree_depth) {
                        offspring_trees.push_back(std::move(new_tree));
                        mutations_performed++;
                        continue;
                    }
                }
            } catch (const std::exception& e) {
                std::cout << "Mutation failed for tree " << i << ": " << e.what() << std::endl;
            }
            
            // If both crossover and mutation fail, create a new random tree
            std::cout << "Creating random tree for position " << i << std::endl;
            offspring_trees.push_back(TreeOperations::createRandomOperatorNode(
                config.max_tree_depth - 1, 0, config, ops, getGen()));
        }
        
        std::cout << "Crossover complete: " << crossovers_performed << " crossovers, "
                << mutations_performed << " mutations performed" << std::endl;
                
        return Individual(std::move(offspring_trees), ops, config);
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
        // auto filePath = "/vol/ecrg-solar/woodj4/fishy-business/data/REIMS.xlsx";
        auto filePath = "/home/woodj/Desktop/fishy-business/data/REIMS.xlsx";
        auto allData = processor.readExcel(filePath, "All data no QC filtering");
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