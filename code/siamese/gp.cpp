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

// Configuration struct
struct GPConfig {
    int n_features = 2080;
    int num_trees = 20;                   // Keep this
    int population_size = 100;            // Increase from 100 to 200 for more diversity
    int generations = 50;                // Keep this
    int elite_size = 5;                   // Reduce from 10 to 5 to prevent overfitting to elite solutions
    float crossover_prob = 0.8f;          // Slightly reduce from 0.8
    float mutation_prob = 0.2f;           // Increase from 0.3 for more exploration
    int tournament_size = 5;              // Reduce from 7 to decrease selection pressure
    float distance_threshold = 0.5f;      // Keep this
    float margin = 1.0f;                  // Increase from 1.0 to create bigger separation
    float fitness_alpha = 0.8f;           // Reduce from 0.9 to put less emphasis on accuracy
    float loss_alpha = 0.2f;              // Increase from 0.1 for better generalization
    float parsimony_coeff = 0.001f;        // Increase from 0.001 to strongly penalize complex trees
    int max_tree_depth = 6;               // Reduce from 6 to prevent overly complex trees
    int batch_size = 64;                  // Reduce from 128 for more frequent updates
    int num_workers = std::thread::hardware_concurrency();
    float dropout_prob = 0.1f;            // Increase from 0.001 for stronger regularization
    float bn_momentum = 0.001f;             // Keep this
    float bn_epsilon = 1e-5f;             // Keep this
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

    std::vector<DataPoint> generatePairs(const std::vector<Instance>& instances, 
                                   size_t pairs_per_instance = 50) {
        std::vector<DataPoint> pairs;
        const size_t expected_total = instances.size() * pairs_per_instance;
        pairs.reserve(expected_total);
        
        std::random_device rd;
        std::mt19937 gen(rd());

        // For each instance, generate exactly pairs_per_instance pairs
        for (size_t i = 0; i < instances.size(); ++i) {
            const Instance& anchor = instances[i];
            
            // We want 25 positive and 25 negative pairs for each instance
            size_t positive_needed = pairs_per_instance / 2;
            size_t negative_needed = pairs_per_instance - positive_needed;
            
            // Collect all possible positive and negative pair candidates
            std::vector<size_t> positive_candidates;
            std::vector<size_t> negative_candidates;
            
            for (size_t j = 0; j < instances.size(); ++j) {
                if (j != i) {  // Avoid self-comparison
                    if (instances[j].label == anchor.label) {
                        positive_candidates.push_back(j);
                    } else {
                        negative_candidates.push_back(j);
                    }
                }
            }

            // Shuffle candidates
            std::shuffle(positive_candidates.begin(), positive_candidates.end(), gen);
            std::shuffle(negative_candidates.begin(), negative_candidates.end(), gen);

            // Generate positive pairs
            for (size_t p = 0; p < positive_needed && p < positive_candidates.size(); ++p) {
                size_t compare_idx = positive_candidates[p % positive_candidates.size()];
                pairs.emplace_back(anchor.features, 
                                instances[compare_idx].features, 
                                1.0f);
            }

            // If we don't have enough positive candidates, make up the difference with negative pairs
            if (positive_candidates.size() < positive_needed) {
                negative_needed += (positive_needed - positive_candidates.size());
            }

            // Generate negative pairs
            for (size_t n = 0; n < negative_needed && n < negative_candidates.size(); ++n) {
                size_t compare_idx = negative_candidates[n % negative_candidates.size()];
                pairs.emplace_back(anchor.features, 
                                instances[compare_idx].features, 
                                0.0f);
            }

            // If we don't have enough total pairs for this instance, resample from available candidates
            size_t current_pairs = std::min(positive_needed, positive_candidates.size()) +
                                std::min(negative_needed, negative_candidates.size());
            
            while (current_pairs < pairs_per_instance) {
                // Pick randomly from available candidates
                std::vector<size_t>& candidates = !negative_candidates.empty() ? negative_candidates : positive_candidates;
                std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
                size_t compare_idx = candidates[dist(gen)];
                float label = (instances[compare_idx].label == anchor.label) ? 1.0f : 0.0f;
                
                pairs.emplace_back(anchor.features, 
                                instances[compare_idx].features, 
                                label);
                current_pairs++;
            }

            if ((i + 1) % 10 == 0 || i == instances.size() - 1) {
                std::cout << "Generated pairs for instance " << (i + 1) << "/" << instances.size()
                        << " (" << pairs.size() << " total pairs)" << std::endl;
            }
        }
        
        // Print final statistics
        size_t positive_pairs = std::count_if(pairs.begin(), pairs.end(),
            [](const DataPoint& p) { return p.label > 0.5f; });
        size_t negative_pairs = pairs.size() - positive_pairs;
        
        std::cout << "\nFinal pair generation statistics:"
                << "\nTotal pairs generated: " << pairs.size() << "/" << expected_total
                << "\nPositive pairs: " << positive_pairs 
                << " (" << (100.0f * positive_pairs / pairs.size()) << "%)"
                << "\nNegative pairs: " << negative_pairs
                << " (" << (100.0f * negative_pairs / pairs.size()) << "%)"
                << "\nAverage pairs per instance: " 
                << (static_cast<float>(pairs.size()) / instances.size()) << std::endl;
        
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
        
            // Shuffle instances before generating pairs
            std::random_device rd;
            std::mt19937 shuffler(rd());
            std::shuffle(instances.begin(), instances.end(), shuffler);
            
            // Generate pairs with shuffled instances
            std::cout << "Generating pairs from shuffled instances..." << std::endl;
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
    virtual std::vector<GPNode*> getAllNodes() = 0;
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

    std::vector<float> evaluate(const std::vector<float>& input) override {
        if (feature_index >= input.size()) {
            std::cerr << "Warning: Feature index " << feature_index 
                      << " out of bounds for input size " << input.size() << std::endl;
            return std::vector<float>{0.0f};
        }
        
        return std::vector<float>{input[feature_index]};
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

    std::vector<GPNode*> getAllNodes() override {
        return {this};
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

    void mutate(std::mt19937& gen, const GPConfig& config) override {
        if (children.empty()) return;
        
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
    const std::vector<std::unique_ptr<GPNode>>& getChildren() const { return children; }
};

class TreeOperations {
public:
    static std::unique_ptr<GPNode> createRandomOperatorNode(
        int max_depth,
        int current_depth,
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        // Create operator node with depth-aware children
        static thread_local std::uniform_int_distribution<int> op_dist(0, 6);
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        // Select operation type
        int op_choice = op_dist(gen);
        std::string op_name;
        bool is_unary = false;
        
        switch (op_choice) {
            case 0: op_name = "add"; break;
            case 1: op_name = "sub"; break;
            case 2: op_name = "div"; break;
            case 3: 
                op_name = "sin"; 
                is_unary = true;
                break;
            case 4: 
                op_name = "cos"; 
                is_unary = true;
                break;
            case 5: 
                op_name = "neg"; 
                is_unary = true;
                break;
            default: op_name = "mul"; break;
        }

        // Create children
        std::vector<std::unique_ptr<GPNode>> children;
        children.push_back(createRandomTree(max_depth, current_depth + 1, config, ops, gen));
        
        // For binary operators, create second child
        if (!is_unary) {
            children.push_back(createRandomTree(max_depth, current_depth + 1, config, ops, gen));
        }

        return std::make_unique<OperatorNode>(op_name, std::move(children), ops);
    }

    static void mutateNode(
        GPNode* node,
        const GPConfig& config,
        std::mt19937& gen
    ) {
        if (!node) return;
        
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        // If this is the root node, ensure it remains an operator
        if (node->isOperator()) {
            auto* op_node = dynamic_cast<OperatorNode*>(node);
            if (op_node) {
                // Potentially mutate operator type while preserving arity
                if (prob_dist(gen) < 0.2f) {
                    bool is_unary = op_node->getChildren().size() == 1;
                    std::vector<std::string> possible_ops;
                    
                    if (is_unary) {
                        possible_ops = {"sin", "cos", "neg"};
                    } else {
                        possible_ops = {"add", "sub", "mul", "div"};
                    }
                    
                    std::uniform_int_distribution<size_t> op_dist(0, possible_ops.size() - 1);
                    // Note: Need to implement setOperator in OperatorNode
                    // op_node->setOperator(possible_ops[op_dist(gen)]);
                }
            }
        }
        
        // Recursively mutate children with decreasing probability
        if (auto* op_node = dynamic_cast<OperatorNode*>(node)) {
            for (const auto& child : op_node->getChildren()) {
                if (child && prob_dist(gen) < config.mutation_prob) {
                    mutateNode(child.get(), config, gen);
                }
            }
        }
    }

private:
    static std::unique_ptr<GPNode> createRandomTree(
        int max_depth,
        int current_depth,
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        static thread_local std::uniform_real_distribution<float> const_dist(-1.0f, 1.0f);

        // Force operator node at root level
        bool force_operator = (current_depth == 0);

        // Force leaf node creation if we're at max depth
        if (current_depth >= max_depth) {
            if (prob_dist(gen) < 0.8f) {
                std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
                return std::make_unique<FeatureNode>(feature_dist(gen));
            } else {
                return std::make_unique<ConstantNode>(const_dist(gen));
            }
        }

        // Early termination with decreasing probability as we go deeper
        // Only allow early termination if we're not at root level
        if (!force_operator) {
            float termination_prob = 0.3f * (1.0f + current_depth / static_cast<float>(max_depth));
            if (prob_dist(gen) < termination_prob) {
                if (prob_dist(gen) < 0.8f) {
                    std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
                    return std::make_unique<FeatureNode>(feature_dist(gen));
                } else {
                    return std::make_unique<ConstantNode>(const_dist(gen));
                }
            }
        }

        // Create operator node with appropriate arity
        return createRandomOperatorNode(max_depth, current_depth, config, ops, gen);
    }
};

// Individual class representing a collection of trees
class Individual {
private:
    std::vector<std::unique_ptr<GPNode>> trees;
    float fitness;
    std::shared_ptr<GPOperations> ops;
    const GPConfig& config;  // Reference to config
    mutable std::mutex eval_mutex;

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

    void mutate(std::mt19937& gen) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& tree : trees) {
            if (dist(gen) < config.mutation_prob) {
                if (tree && tree->isOperator()) {
                    TreeOperations::mutateNode(tree.get(), config, gen);
                } else {
                    // If somehow we got a leaf node at root, replace it
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
    
    std::mt19937& getGen() {
        return rng;  // Now returns reference to thread-local RNG
    }

    std::unique_ptr<GPNode> createRandomOperatorNode(int max_depth, int current_depth) {
        static thread_local std::uniform_int_distribution<int> op_dist(0, 6);
        
        int op_choice = op_dist(rng);
        std::string op_name;
        bool is_unary = false;
        
        switch (op_choice) {
            case 0: op_name = "add"; break;
            case 1: op_name = "sub"; break;
            case 2: op_name = "div"; break;
            case 3: 
                op_name = "sin"; 
                is_unary = true;
                break;
            case 4: 
                op_name = "cos"; 
                is_unary = true;
                break;
            case 5: 
                op_name = "neg"; 
                is_unary = true;
                break;
            default: op_name = "mul"; break;
        }

        std::vector<std::unique_ptr<GPNode>> children;
        children.push_back(createRandomTree(max_depth, current_depth + 1));
        if (!is_unary) {
            children.push_back(createRandomTree(max_depth, current_depth + 1));
        }

        return std::make_unique<OperatorNode>(op_name, std::move(children), ops);
    }

    Individual createRandomIndividual() {
        std::vector<std::unique_ptr<GPNode>> trees;
        for (int i = 0; i < config.num_trees; ++i) {
            trees.push_back(createRandomOperatorNode(config.max_tree_depth, 0));
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

    std::unique_ptr<GPNode> enforceMaxDepth(std::unique_ptr<GPNode> tree, int max_depth) {
        if (!tree) return nullptr;
        
        if (tree->depth() > max_depth) {
            // Replace with a random leaf node if depth exceeds maximum
            std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
            if (prob_dist(rng) < 0.8) {
                std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
                return std::make_unique<FeatureNode>(feature_dist(rng));
            } else {
                std::uniform_real_distribution<float> const_dist(-1.0f, 1.0f);
                return std::make_unique<ConstantNode>(const_dist(rng));
            }
        }
        return tree;
    }

    std::unique_ptr<GPNode> createRandomTree(int max_depth, int current_depth = 0) {
        // Thread-local distributions
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        static thread_local std::uniform_int_distribution<int> op_dist(0, 6);
        static thread_local std::uniform_real_distribution<float> const_dist(-1.0f, 1.0f);

        // At root level (current_depth == 0), always create an operator node
        bool force_operator = (current_depth == 0);

        // Force leaf node creation if we're at max depth
        if (current_depth >= max_depth) {
            if (prob_dist(rng) < 0.8) {
                std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
                return std::make_unique<FeatureNode>(feature_dist(rng));
            } else {
                return std::make_unique<ConstantNode>(const_dist(rng));
            }
        }

        // Early termination with decreasing probability as we go deeper
        // Only allow early termination if we're not at root level
        if (!force_operator) {
            float termination_prob = 0.3f * (1.0f + current_depth / static_cast<float>(max_depth));
            if (prob_dist(rng) < termination_prob) {
                if (prob_dist(rng) < 0.8) {
                    std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
                    return std::make_unique<FeatureNode>(feature_dist(rng));
                } else {
                    return std::make_unique<ConstantNode>(const_dist(rng));
                }
            }
        }

        // Create operator node with depth-aware children
        std::vector<std::unique_ptr<GPNode>> children;
        
        // For unary operators (sin, cos, neg), create one child
        // For binary operators (add, sub, mul, div), create two children
        int op_choice = op_dist(rng);
        std::string op_name;
        bool is_unary = false;
        
        switch (op_choice) {
            case 0: op_name = "add"; break;
            case 1: op_name = "sub"; break;
            case 2: op_name = "div"; break;
            case 3: 
                op_name = "sin"; 
                is_unary = true;
                break;
            case 4: 
                op_name = "cos"; 
                is_unary = true;
                break;
            case 5: 
                op_name = "neg"; 
                is_unary = true;
                break;
            default: op_name = "mul"; break;
        }

        // Always create at least one child
        children.push_back(createRandomTree(max_depth, current_depth + 1));
        
        // For binary operators, create second child
        if (!is_unary) {
            children.push_back(createRandomTree(max_depth, current_depth + 1));
        }

        return std::make_unique<OperatorNode>(op_name, std::move(children), ops);
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
        
        float avg_similar_dist = 0.0f;
        float avg_dissimilar_dist = 0.0f;
        int similar_count = 0;
        int dissimilar_count = 0;
        
        for (size_t i = 0; i < data.size(); i += config.batch_size) {
            size_t batch_end = std::min(i + config.batch_size, data.size());
            std::vector<DataPoint> batch(data.begin() + i, data.begin() + batch_end);
            
            for (const auto& point : batch) {
                auto anchor_output = ind.evaluate(point.anchor);
                auto compare_output = ind.evaluate(point.compare);
                
                float distance = calculateDistance(anchor_output, compare_output);
                bool prediction = distance < config.distance_threshold;
                bool actual = point.label > 0.5f;
                
                if (prediction == actual) {
                    correct_predictions++;
                }
                total_pairs++;
            }
        }
        
        // Calculate averages
        if (similar_count > 0) avg_similar_dist /= similar_count;
        if (dissimilar_count > 0) avg_dissimilar_dist /= dissimilar_count;
        
        float accuracy = static_cast<float>(correct_predictions) / total_pairs;
        return accuracy;
    }

     // Helper method to run a single tournament
    Individual& runTournament(const std::vector<size_t>& available_indices) {
        if (available_indices.empty()) {
            throw std::runtime_error("No individuals available for tournament");
        }

        // Randomly select tournament_size individuals
        std::vector<size_t> tournament_candidates;
        tournament_candidates.reserve(config.tournament_size);
        
        auto& local_gen = getGen();
        std::uniform_int_distribution<size_t> idx_dist(0, available_indices.size() - 1);
        
        // Fill tournament pool
        for (int i = 0; i < config.tournament_size && i < available_indices.size(); ++i) {
            size_t random_idx = idx_dist(local_gen);
            tournament_candidates.push_back(available_indices[random_idx]);
        }

        // Find the best individual in the tournament
        size_t best_idx = tournament_candidates[0];
        float best_fitness = population[best_idx].getFitness();

        for (size_t idx : tournament_candidates) {
            float current_fitness = population[idx].getFitness();
            if (current_fitness < best_fitness) { // Lower fitness is better
                best_fitness = current_fitness;
                best_idx = idx;
            }
        }

        return population[best_idx];
    }

    void evaluatePopulation(const std::vector<DataPoint>& trainData) {
         // Add depth monitoring
        int max_depth_seen = 0;
        int total_depth = 0;
        int valid_trees = 0;
        
        for (const auto& individual : population) {
            for (int i = 0; i < config.num_trees; ++i) {
                const auto& tree = individual.getTree(i);
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
        if (valid_trees > 0) {
            float avg_depth = static_cast<float>(total_depth) / valid_trees;
            std::cout << "Depth statistics:"
                      << "\n  Max depth: " << max_depth_seen
                      << "\n  Average depth: " << avg_depth
                      << "\n  Valid trees: " << valid_trees << "/" 
                      << (population.size() * config.num_trees) << std::endl;
        }

        const size_t num_workers = std::min((size_t)config.num_workers, population.size());
        const size_t batch_size = (population.size() + num_workers - 1) / num_workers;

        // Add before evaluation
        size_t total_nodes = 0;
        for (const auto& individual : population) {
            total_nodes += individual.totalSize();
        }
        std::cout << "Average tree size: " << (float)total_nodes / population.size() << std::endl;

        std::cout << "Population size: " << population.size() << std::endl;
        
        std::vector<std::future<void>> futures;
        
        for (size_t i = 0; i < population.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, population.size());
            futures.push_back(std::async(std::launch::async, [this, i, end, &trainData]() {
                for (size_t j = i; j < end; ++j) {
                    float fitness = evaluateIndividual(population[j], trainData);
                    population[j].setFitness(fitness);
                }
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
    }

    std::unique_ptr<GPNode> validateAndFixDepth(std::unique_ptr<GPNode> tree) {
        if (!tree) return nullptr;
        
        // Check if tree depth is valid
        if (!tree->isValidDepth()) {
            std::cerr << "Invalid tree depth detected, truncating..." << std::endl;
            return enforceMaxDepth(std::move(tree), config.max_tree_depth);
        }
        
        return tree;
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

        // Setup file for CSV logging
        std::ofstream timing_log;
        timing_log.open("gp_timing.csv");
        timing_log << "Generation,Operation,Total_Time,Calls,Average_Time\n";

        for (int generation = 0; generation < config.generations; ++generation) {
            profiler.startGeneration();
            std::cout << "\nStarting generation " << generation << std::endl;
            
            // Profile population evaluation
            auto eval_start = std::chrono::high_resolution_clock::now();
            evaluatePopulation(trainData);
            auto eval_end = std::chrono::high_resolution_clock::now();
            profiler.recordTime("Population_Evaluation", 
                std::chrono::duration<double>(eval_end - eval_start).count());
            
            // Profile fitness statistics calculation
            auto stats_start = std::chrono::high_resolution_clock::now();
            Individual* best_individual = nullptr;
            float gen_best_fitness = std::numeric_limits<float>::max();
            float gen_avg_fitness = 0.0f;
            
            for (auto& individual : population) {
                float fitness = individual.getFitness();
                gen_avg_fitness += fitness;
                if (fitness < gen_best_fitness) {
                    gen_best_fitness = fitness;
                    best_individual = &individual;
                }
            }
            gen_avg_fitness /= population.size();
            auto stats_end = std::chrono::high_resolution_clock::now();
            profiler.recordTime("Fitness_Statistics", 
                std::chrono::duration<double>(stats_end - stats_start).count());

            // Profile new population creation
            auto new_pop_start = std::chrono::high_resolution_clock::now();
            std::vector<Individual> new_population;
            new_population.reserve(config.population_size);

            // Profile elitism
            auto elitism_start = std::chrono::high_resolution_clock::now();
            std::vector<size_t> indices(population.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                    [this](size_t a, size_t b) { 
                        return population[a].getFitness() < population[b].getFitness(); 
                    });

            for (int i = 0; i < config.elite_size && i < population.size(); ++i) {
                new_population.push_back(Individual(population[indices[i]]));
            }
            auto elitism_end = std::chrono::high_resolution_clock::now();
            profiler.recordTime("Elitism", 
                std::chrono::duration<double>(elitism_end - elitism_start).count());

            // Profile crossover and mutation
            double total_crossover_time = 0.0;
            double total_mutation_time = 0.0;
            int crossover_count = 0;
            int mutation_count = 0;

            auto breeding_start = std::chrono::high_resolution_clock::now();
            size_t remaining_slots = config.population_size - new_population.size();
            size_t pairs_needed = (remaining_slots + 1) / 2;  // Round up division

            // Profile parent selection
            auto selection_start = std::chrono::high_resolution_clock::now();
            auto selected_pairs = selectParents(pairs_needed);
            auto selection_end = std::chrono::high_resolution_clock::now();
            profiler.recordTime("Parent_Selection", 
                std::chrono::duration<double>(selection_end - selection_start).count());

            // Process all pairs
            for (const auto& [parent1, parent2] : selected_pairs) {
                if (new_population.size() >= config.population_size) {
                    break;
                }

                auto crossover_start = std::chrono::high_resolution_clock::now();
                Individual offspring = crossover(*parent1, *parent2);
                auto crossover_end = std::chrono::high_resolution_clock::now();
                total_crossover_time += std::chrono::duration<double>(
                    crossover_end - crossover_start).count();
                crossover_count++;

                auto& local_gen = getGen();
                std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
                if (prob_dist(local_gen) < config.mutation_prob) {
                    auto mutation_start = std::chrono::high_resolution_clock::now();
                    offspring.mutate(local_gen);  // Updated to match new signature
                    auto mutation_end = std::chrono::high_resolution_clock::now();
                    total_mutation_time += std::chrono::duration<double>(
                        mutation_end - mutation_start).count();
                    mutation_count++;
                }

                new_population.push_back(std::move(offspring));

                if (new_population.size() < config.population_size) {
                    auto crossover_start2 = std::chrono::high_resolution_clock::now();
                    Individual second_offspring = crossover(*parent2, *parent1);
                    auto crossover_end2 = std::chrono::high_resolution_clock::now();
                    total_crossover_time += std::chrono::duration<double>(
                        crossover_end2 - crossover_start2).count();
                    crossover_count++;
                    
                    if (prob_dist(local_gen) < config.mutation_prob) {
                        auto mutation_start = std::chrono::high_resolution_clock::now();
                        second_offspring.mutate(local_gen);  // Updated to match new signature
                        auto mutation_end = std::chrono::high_resolution_clock::now();
                        total_mutation_time += std::chrono::duration<double>(
                            mutation_end - mutation_start).count();
                        mutation_count++;
                    }
                    
                    new_population.push_back(std::move(second_offspring));
                }
            }

            // Profile population completion
            auto completion_start = std::chrono::high_resolution_clock::now();

            // Fill any remaining slots with clones of the best individuals
            while (new_population.size() < config.population_size) {
                size_t idx = new_population.size() % population.size();
                new_population.push_back(Individual(population[indices[idx]]));
            }

            // Verify population size
            if (new_population.size() != config.population_size) {
                std::cerr << "Error: Population size mismatch. Expected " << config.population_size 
                        << " but got " << new_population.size() << std::endl;
                // Adjust population size if needed
                while (new_population.size() > config.population_size) {
                    new_population.pop_back();
                }
                while (new_population.size() < config.population_size) {
                    new_population.push_back(Individual(population[0])); // Clone best individual
                }
            }

            auto breeding_end = std::chrono::high_resolution_clock::now();

            // Record breeding statistics
            profiler.recordTime("Total_Breeding", 
                std::chrono::duration<double>(breeding_end - breeding_start).count());
            if (crossover_count > 0) {
                profiler.recordTime("Average_Crossover", total_crossover_time / crossover_count);
            }
            if (mutation_count > 0) {
                profiler.recordTime("Average_Mutation", total_mutation_time / mutation_count);
            }
            profiler.recordTime("Population_Completion", 
                std::chrono::duration<double>(breeding_end - completion_start).count());

            // Profile accuracy calculations
            auto accuracy_start = std::chrono::high_resolution_clock::now();
            float train_accuracy = calculateAccuracy(population[indices[0]], trainData);
            ops->setTraining(false);
            float val_accuracy = calculateAccuracy(population[indices[0]], valData);
            ops->setTraining(true);
            auto accuracy_end = std::chrono::high_resolution_clock::now();
            profiler.recordTime("Accuracy_Calculation", 
                std::chrono::duration<double>(accuracy_end - accuracy_start).count());

            // Output generation statistics
            std::cout << "Generation " << generation 
                    << "\n  Best Fitness: " << gen_best_fitness
                    << "\n  Avg Fitness: " << gen_avg_fitness
                    << "\n  Training Accuracy: " << train_accuracy * 100.0f << "%"
                    << "\n  Validation Accuracy: " << val_accuracy * 100.0f << "%"
                    << "\n  Population Size: " << new_population.size()
                    << "\n  Generation Time: " << profiler.getGenerationTime() << "s"
                    << std::endl;

            // Print detailed profiling statistics and save to CSV
            for (const auto& [operation, data] : profiler.getTimings()) {
                timing_log << generation << ","
                        << operation << ","
                        << data.total_time << ","
                        << data.calls << ","
                        << (data.total_time / data.calls) << "\n";
            }

            // Print current generation profiling statistics
            // profiler.printStatistics(generation);

            // Check for improvement
            if (gen_best_fitness < best_fitness) {
                best_fitness = gen_best_fitness;
                generations_without_improvement = 0;
            } else {
                generations_without_improvement++;
            }

            // Replace old population
            population = std::move(new_population);
        }
        
        timing_log.close();
    }
    
    
    Individual crossover(const Individual& parent1, const Individual& parent2) {
        std::vector<std::unique_ptr<GPNode>> offspring_trees;
        offspring_trees.reserve(config.num_trees);
        
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        for (int i = 0; i < config.num_trees; ++i) {
            if (dist(rng) < config.crossover_prob) {
                auto parent1_tree = parent1.getTree(i).get();
                auto parent2_tree = parent2.getTree(i).get();
                
                auto nodes1 = parent1_tree->getAllNodes();
                auto nodes2 = parent2_tree->getAllNodes();
                
                if (nodes1.empty() || nodes2.empty()) {
                    offspring_trees.push_back(createRandomOperatorNode(config.max_tree_depth, 0));
                    continue;
                }
                
                // Filter nodes to get only operator nodes from parent2 when selecting root replacement
                std::vector<GPNode*> operator_nodes2;
                for (auto* node : nodes2) {
                    if (node->isOperator()) {
                        operator_nodes2.push_back(node);
                    }
                }
                
                // If no operator nodes found in parent2, create a new random operator node
                if (operator_nodes2.empty()) {
                    offspring_trees.push_back(createRandomOperatorNode(config.max_tree_depth, 0));
                    continue;
                }
                
                // Select random crossover points
                std::uniform_int_distribution<size_t> node_dist1(0, nodes1.size() - 1);
                std::uniform_int_distribution<size_t> op_dist2(0, operator_nodes2.size() - 1);
                
                int max_attempts = 10;
                bool valid_crossover = false;
                std::unique_ptr<GPNode> new_tree;
                
                while (max_attempts-- > 0) {
                    GPNode* subtree1 = nodes1[node_dist1(rng)];
                    GPNode* subtree2 = operator_nodes2[op_dist2(rng)];
                    
                    try {
                        auto new_tree1 = parent1.getTree(i)->clone();
                        auto subtree2_clone = subtree2->clone();
                        
                        int parent_depth = parent1_tree->depth();
                        int subtree1_depth = subtree1->depth();
                        int subtree2_depth = subtree2->depth();
                        
                        int new_depth = parent_depth - subtree1_depth + subtree2_depth;
                        
                        if (new_depth <= config.max_tree_depth) {
                            // For root node replacement, ensure we're using an operator node
                            if (subtree1 == parent1_tree) {
                                if (subtree2->isOperator()) {
                                    new_tree = std::move(subtree2_clone);
                                    valid_crossover = true;
                                    break;
                                }
                            } else {
                                new_tree1->replaceSubtree(subtree1, std::move(subtree2_clone));
                                new_tree = std::move(new_tree1);
                                valid_crossover = true;
                                break;
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Crossover attempt failed: " << e.what() << std::endl;
                    }
                }
                
                if (valid_crossover) {
                    offspring_trees.push_back(std::move(new_tree));
                } else {
                    // Fallback to creating a new random operator node
                    offspring_trees.push_back(createRandomOperatorNode(config.max_tree_depth, 0));
                }
            } else {
                offspring_trees.push_back(parent1.getTree(i)->clone());
            }
        }

        // Validate all trees have operator nodes at root
        for (auto& tree : offspring_trees) {
            if (!tree || !tree->isOperator()) {
                tree = createRandomOperatorNode(config.max_tree_depth, 0);
            }
        }
        
        // Pass config along with trees and ops
        return Individual(std::move(offspring_trees), ops, config);
    }

     void mutateNode(GPNode* node, std::mt19937& gen, const GPConfig& config) {
        if (!node) return;
        
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        // If this is the root node, ensure it remains an operator
        if (node->isOperator()) {
            auto* op_node = dynamic_cast<OperatorNode*>(node);
            if (op_node) {
                // Potentially mutate operator type while preserving arity
                if (prob_dist(gen) < 0.2f) {
                    bool is_unary = op_node->getChildren().size() == 1;
                    std::vector<std::string> possible_ops;
                    
                    if (is_unary) {
                        possible_ops = {"sin", "cos", "neg"};
                    } else {
                        possible_ops = {"add", "sub", "mul", "div"};
                    }
                    
                    std::uniform_int_distribution<size_t> op_dist(0, possible_ops.size() - 1);
                    // Note: You'll need to add a method to OperatorNode to change its operator
                    // op_node->setOperator(possible_ops[op_dist(gen)]);
                }
            }
        }
        
        // Recursively mutate children
        if (auto* op_node = dynamic_cast<OperatorNode*>(node)) {
            for (auto& child : op_node->getChildren()) {
                if (child && prob_dist(gen) < config.mutation_prob) {
                    mutateNode(child.get(), gen, config);
                }
            }
        }
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