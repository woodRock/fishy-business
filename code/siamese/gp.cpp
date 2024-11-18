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
#include <iomanip>
#include <atomic>
#include <mutex>

// Configuration struct
struct GPConfig {
    int n_features = 2080;
    int num_trees = 20;                   // Increased from 20 to 30 for better ensemble
    int population_size = 100;            // Doubled population size for more diversity
    int generations = 100;                // Doubled to allow more evolution time
    int elite_size = 10;                  // Increased to preserve good solutions
    float crossover_prob = 0.8f;         // Slightly increased for more genetic material exchange
    float mutation_prob = 0.2f;          // Increased for better exploration
    int tournament_size = 7;             // Increased for stronger selection pressure
    int classifier_trees = 20;     // Number of trees for voting
    float decision_threshold = 0.5f;  // Classification threshold
    float feature_scale = 0.01f;   // Scale factor for feature normalization
    int min_votes = 5;           // Minimum votes for positive classification
    float distance_threshold = 1.0f;      // Reduced to be more selective
    float margin = 1.0f;      
    float initial_distance_threshold = 1.0f;
    float threshold_adaptation_rate = 0.01f;
    float threshold_min = 0.1f;
    float threshold_max = 2.0f;
    int threshold_window_size = 100;             // Increased margin for better separation
    float fitness_alpha = 0.7f;           // Reduced to balance accuracy vs. complexity
    float loss_alpha = 0.3f;              // Increased to focus more on loss reduction
    float parsimony_coeff = 0.0005f;      // Reduced to allow more complex solutions
    int max_tree_depth = 6;               // Increased for more expressive trees
    int min_tree_depth = 3;               // Kept the same
    int batch_size = 32;                  // Reduced for more frequent updates
    int num_workers = std::thread::hardware_concurrency();
    float dropout_prob = 0.1f;           // Increased for stronger regularization
    float bn_momentum = 0.001f;            // Reduced  faster adaptation
    float bn_epsilon = 1e-6f;             // Reduced for more precise normalization
};

inline std::mt19937& getThreadLocalRNG() {
    static thread_local std::mt19937 rng{std::random_device{}()};
    return rng;
}

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
        // If either input is empty, return zero
        if (a.empty() || b.empty()) {
            return std::vector<float>{0.0f};
        }

        // For scalar broadcast, always return scalar result
        if (a.size() == 1 || b.size() == 1) {
            const float a_val = a[0];
            const float b_val = b[0];
            
            if (std::isinf(a_val) || std::isinf(b_val) || std::isnan(a_val) || std::isnan(b_val)) {
                return std::vector<float>{0.0f};
            }
            
            return std::vector<float>{a_val + b_val};
        }

        // Otherwise, sizes must match exactly
        if (a.size() != b.size()) {
            std::cerr << "Vector addition: Mismatched sizes (a: " << a.size() 
                     << ", b: " << b.size() << ")" << std::endl;
            return std::vector<float>{0.0f};
        }

        std::vector<float> result(1);  // Always return scalar
        float sum = 0.0f;
        
        // Sum all elements
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::isinf(a[i]) || std::isinf(b[i]) || std::isnan(a[i]) || std::isnan(b[i])) {
                return std::vector<float>{0.0f};
            }
            sum += a[i] + b[i];
        }
        
        result[0] = sum / static_cast<float>(a.size());  // Return average
        return result;
    }

    inline std::vector<float> subtract(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty()) {
            return std::vector<float>{0.0f};
        }

        // For scalar broadcast, always return scalar result
        if (a.size() == 1 || b.size() == 1) {
            const float a_val = a[0];
            const float b_val = b[0];
            
            if (std::isinf(a_val) || std::isinf(b_val) || std::isnan(a_val) || std::isnan(b_val)) {
                return std::vector<float>{0.0f};
            }
            
            return std::vector<float>{a_val - b_val};
        }

        if (a.size() != b.size()) {
            std::cerr << "Vector subtraction: Mismatched sizes (a: " << a.size() 
                     << ", b: " << b.size() << ")" << std::endl;
            return std::vector<float>{0.0f};
        }

        std::vector<float> result(1);
        float diff = 0.0f;
        
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::isinf(a[i]) || std::isinf(b[i]) || std::isnan(a[i]) || std::isnan(b[i])) {
                return std::vector<float>{0.0f};
            }
            diff += a[i] - b[i];
        }
        
        result[0] = diff / static_cast<float>(a.size());
        return result;
    }

    inline std::vector<float> multiply(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty()) {
            return std::vector<float>{0.0f};
        }

        // For scalar broadcast, always return scalar result
        if (a.size() == 1 || b.size() == 1) {
            const float a_val = a[0];
            const float b_val = b[0];
            
            if (std::isinf(a_val) || std::isinf(b_val) || std::isnan(a_val) || std::isnan(b_val)) {
                return std::vector<float>{0.0f};
            }
            
            return std::vector<float>{a_val * b_val};
        }

        if (a.size() != b.size()) {
            std::cerr << "Vector multiplication: Mismatched sizes (a: " << a.size() 
                     << ", b: " << b.size() << ")" << std::endl;
            return std::vector<float>{0.0f};
        }

        std::vector<float> result(1);
        float prod = 0.0f;
        
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::isinf(a[i]) || std::isinf(b[i]) || std::isnan(a[i]) || std::isnan(b[i])) {
                return std::vector<float>{0.0f};
            }
            prod += a[i] * b[i];
        }
        
        result[0] = prod / static_cast<float>(a.size());
        return result;
    }

    inline std::vector<float> divide(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty()) {
            return std::vector<float>{0.0f};
        }

        // For scalar broadcast, always return scalar result
        if (a.size() == 1 || b.size() == 1) {
            const float a_val = a[0];
            const float b_val = b[0];
            
            if (std::isinf(a_val) || std::isinf(b_val) || std::isnan(a_val) || std::isnan(b_val)) {
                return std::vector<float>{0.0f};
            }
            
            return std::vector<float>{std::abs(b_val) < 1e-10f ? a_val : a_val / (b_val + 1e-10f)};
        }

        if (a.size() != b.size()) {
            std::cerr << "Vector division: Mismatched sizes (a: " << a.size() 
                     << ", b: " << b.size() << ")" << std::endl;
            return std::vector<float>{0.0f};
        }

        std::vector<float> result(1);
        float sum = 0.0f;
        float count = 0.0f;
        
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::isinf(a[i]) || std::isinf(b[i]) || std::isnan(a[i]) || std::isnan(b[i])) {
                continue;
            }
            if (std::abs(b[i]) >= 1e-10f) {
                sum += a[i] / (b[i] + 1e-10f);
                count += 1.0f;
            } else {
                sum += a[i];
                count += 1.0f;
            }
        }
        
        result[0] = count > 0.0f ? sum / count : 0.0f;
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
        operations = {
            {"add", [this](const auto& inputs, size_t id) { return vec_ops::add(inputs[0], inputs[1]); }},
            {"sub", [this](const auto& inputs, size_t id) { return vec_ops::subtract(inputs[0], inputs[1]); }},
            {"mul", [this](const auto& inputs, size_t id) { return vec_ops::multiply(inputs[0], inputs[1]); }},
            {"div", [this](const auto& inputs, size_t id) { return protectedDiv(inputs[0], inputs[1]);}},
            {"sin", [this](const auto& inputs, size_t id) { return applySin(inputs[0]); }},
            {"cos", [this](const auto& inputs, size_t id) { return applyCos(inputs[0]); }},
            {"neg", [this](const auto& inputs, size_t id) { return applyNeg(inputs[0]); }}
        };
        // Initialize batch normalizations for each operation type
        initializeBatchNorms(batch_norm_momentum, batch_norm_epsilon);
    }

    // Disable copying
    GPOperations(const GPOperations&) = delete;
    GPOperations& operator=(const GPOperations&) = delete;

    bool isTraining() const { return training; }  // Add this accessor

    // Main evaluation interface
    // In GPOperations class, modify the evaluate method:
    std::vector<float> evaluate(const std::string& op_name,
                    const std::vector<std::vector<float>>& inputs,
                    size_t node_id) {
        std::lock_guard<std::mutex> lock(mutex);
        
        try {
            // Input validation
            if (op_name.empty() || inputs.empty()) {
                return std::vector<float>{0.0f};
            }

            // Check input sizes
            if (inputs.size() < 1 || inputs[0].empty()) {
                return std::vector<float>{0.0f};
            }

            const size_t input_size = inputs[0].size();
            for (const auto& input : inputs) {
                if (input.size() != input_size) {
                    return std::vector<float>{0.0f};
                }
            }

            // Validate operation exists
            auto op_it = operations.find(op_name);
            if (op_it == operations.end()) {
                return std::vector<float>{0.0f};
            }

            // Track evaluation state
            current_eval_id = node_id;

            // Perform operation with exception handling
            std::vector<float> result;
            try {
                result = op_it->second(inputs, node_id);
            } catch (...) {
                return std::vector<float>{0.0f};
            }

            // Validate result
            if (result.empty() || result.size() != input_size) {
                return std::vector<float>{0.0f};
            }

            // Apply batch normalization
            auto bn_it = batch_norms.find(op_name);
            if (bn_it != batch_norms.end() && bn_it->second) {
                try {
                    result = bn_it->second->operator()(result);
                } catch (const std::exception& e) {
                    std::cerr << "BatchNorm error: " << e.what() << std::endl;
                    // Continue with unnormalized result if BatchNorm fails
                }
            }

            // Apply dropout if enabled
            if (training && dropout_prob > 0.0f) {
                static thread_local std::mt19937 gen(std::random_device{}());
                std::bernoulli_distribution dropout(dropout_prob);
                
                for (auto& val : result) {
                    if (dropout(gen)) {
                        val = 0.0f;
                    }
                }
            }

            return result;

        } catch (...) {
            return std::vector<float>{0.0f};
        }
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
    std::unordered_map<std::string, std::function<std::vector<float>(const std::vector<std::vector<float>>&, size_t)>> operations;
    std::mutex op_mutex; // Add this line to declare the mutex

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

    std::vector<float> safeAdd(const std::vector<float>& x, const std::vector<float>& y) {
        auto a = x;
        auto b = y;
        if (y.size() > 1) {
            b = {y[0]};
        }
        return vec_ops::add(a,b);
    }

    std::vector<float> safeSubtract(const std::vector<float>& x, const std::vector<float>& y) {
        auto a = x;
        auto b = y;
        if (y.size() > 1) {
            b = {y[0]};
        }
        return vec_ops::subtract(a,b);
    }

    std::vector<float> safeMultiply(const std::vector<float>& x, const std::vector<float>& y) {
        auto a = x;
        auto b = y;
        if (y.size() > 1) {
            b = {y[0]};
        }
        return vec_ops::multiply(a,b);
    }

    std::vector<float> protectedDiv(const std::vector<float>& x,
                                  const std::vector<float>& y) {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            if (std::isinf(x[i]) || std::isinf(y[i]) || std::isnan(x[i]) || std::isnan(y[i])) {
                std::cerr << "Overflow/Underflow detected in division" << std::endl;
                return std::vector<float>{0.0f}; // Handle overflow/underflow
            }
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
        try {
            // Input validation
            if (input.empty()) {
                std::cerr << "Empty input vector in FeatureNode" << std::endl;
                return std::vector<float>{0.0f};
            }
            
            // Index bounds check
            if (feature_index >= input.size()) {
                std::cerr << "Feature index " << feature_index 
                        << " out of bounds for input size " << input.size() << std::endl;
                return std::vector<float>{0.0f};
            }
            
            // Value validity check
            if (!std::isfinite(input[feature_index])) {
                std::cerr << "Non-finite input value at index " << feature_index << std::endl;
                return std::vector<float>{0.0f};
            }
            
            return std::vector<float>{input[feature_index]};
            
        } catch (...) {
            std::cerr << "Unexpected error in FeatureNode::evaluate" << std::endl;
            return std::vector<float>{0.0f};
        }
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
        try {
            // Value validity check
            if (!std::isfinite(value)) {
                std::cerr << "Non-finite constant value" << std::endl;
                return std::vector<float>{0.0f};
            }
            
            return std::vector<float>{value};
            
        } catch (...) {
            std::cerr << "Unexpected error in ConstantNode::evaluate" << std::endl;
            return std::vector<float>{0.0f};
        }
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
        if (input.empty()) {
            return std::vector<float>{0.0f};
        }
        
        if (children.empty()) {
            return std::vector<float>{0.0f};
        }
        
        try {
            // Evaluate children with strict error checking
            std::vector<std::vector<float>> child_results;
            child_results.reserve(children.size());
            
            for (const auto& child : children) {
                if (!child) {
                    return std::vector<float>{0.0f};
                }
                
                // Safe evaluation
                std::vector<float> result;
                try {
                    result = child->evaluate(input);
                } catch (...) {
                    return std::vector<float>{0.0f};
                }
                
                // Validate result
                if (result.empty()) {
                    return std::vector<float>{0.0f};
                }
                
                // Ensure each child returns scalar
                if (result.size() > 1) {
                    result = std::vector<float>{result[0]};
                }
                
                child_results.push_back(std::move(result));
            }
            
            // Safe operator evaluation with error handling
            try {
                auto result = ops->evaluate(op_name, child_results, node_id);
                
                // Ensure final result is scalar
                if (result.empty()) {
                    return std::vector<float>{0.0f};
                }
                if (result.size() > 1) {
                    result = std::vector<float>{result[0]};
                }
                
                return result;
            } catch (...) {
                return std::vector<float>{0.0f};
            }
            
        } catch (...) {
            return std::vector<float>{0.0f};
        }
    }

    void replaceChild(GPNode* old_child, std::unique_ptr<GPNode> new_child) {
        for (auto& child : children) {
            if (child.get() == old_child) {
                child = std::move(new_child);
                return;
            }
        }
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
    std::shared_ptr<GPOperations> getOperations() const { return ops; }

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

    static void validateTreeOutputs(GPNode* node, const GPConfig& config) {
        if (!node) {
            throw std::runtime_error("Null node in validation");
        }
        
        try {
            // Test evaluation with dummy input
            std::vector<float> test_input(config.n_features, 0.0f);
            auto result = node->evaluate(test_input);
            
            // Every node must output a scalar
            if (result.size() != 1) {
                throw std::runtime_error("Invalid node output size: " + std::to_string(result.size()));
            }

            // Check for NaN/Inf
            if (!std::isfinite(result[0])) {
                throw std::runtime_error("Node produced non-finite output");
            }
            
            // Validate children of operator nodes
            if (auto* op_node = dynamic_cast<OperatorNode*>(node)) {
                for (const auto& child : op_node->getChildren()) {
                    if (!child) {
                        throw std::runtime_error("Null child in operator node");
                    }
                    validateTreeOutputs(child.get(), config);
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Tree validation failed: " + std::string(e.what()));
        }
    }

    void validateTreeStructure(GPNode* node, int depth = 0) {
        if (!node) {
            throw std::runtime_error("Null node encountered");
        }

        // Check basic properties
        if (depth > 1000) {
            throw std::runtime_error("Tree exceeds maximum depth");
        }

        // If it's an operator node, validate its children
        if (auto* op_node = dynamic_cast<OperatorNode*>(node)) {
            const auto& children = op_node->getChildren();
            
            // Validate operator type matches number of children
            bool is_unary = (op_node->getOperatorName() == "sin" || 
                            op_node->getOperatorName() == "cos" || 
                            op_node->getOperatorName() == "neg");
            
            if (is_unary && children.size() != 1) {
                throw std::runtime_error("Unary operator with incorrect number of children: " + 
                                    op_node->getOperatorName());
            }
            
            if (!is_unary && children.size() != 2) {
                throw std::runtime_error("Binary operator with incorrect number of children: " + 
                                    op_node->getOperatorName());
            }

            // Recursively validate all children
            for (const auto& child : children) {
                if (!child) {
                    throw std::runtime_error("Null child in operator node");
                }
                validateTreeStructure(child.get(), depth + 1);
            }
        }
    }

public:

    // Helper function to ensure minimum depth
    static bool hasMinimumDepth(const GPNode* node, int min_depth) {
        if (!node) return false;
        if (min_depth <= 1) return true;
        
        if (const auto* op_node = dynamic_cast<const OperatorNode*>(node)) {
            const auto& children = op_node->getChildren();
            for (const auto& child : children) {
                if (hasMinimumDepth(child.get(), min_depth - 1)) {
                    return true;
                }
            }
        }
        return false;
    }

    // Modified to enforce minimum depth
    static std::unique_ptr<GPNode> createRandomOperatorNode(
        int max_depth,
        int current_depth,
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        // Calculate required depth to meet minimum
        int depth_needed = config.min_tree_depth - current_depth;
        
        // If we need more depth than available, create full binary tree
        if (depth_needed >= max_depth - current_depth) {
            return createFullBinaryTree(depth_needed, config, ops, gen);
        }

        static const std::vector<std::string> binary_ops = {"add", "mul", "sub", "div"};
        std::uniform_int_distribution<size_t> op_dist(0, binary_ops.size() - 1);
        
        // Always create binary operator to ensure depth potential
        std::vector<std::unique_ptr<GPNode>> children;
        
        // First child must maintain minimum depth
        auto first_child = createRandomOperatorNode(
            max_depth - 1,
            current_depth + 1,
            config,
            ops,
            gen
        );
        if (!first_child) return nullptr;
        children.push_back(std::move(first_child));

        // Second child can be simpler if minimum depth is already met
        auto second_child = (depth_needed > 1) ?
            createRandomOperatorNode(max_depth - 1, current_depth + 1, config, ops, gen) :
            createLeafNode(config, gen);
        if (!second_child) return nullptr;
        children.push_back(std::move(second_child));

        return std::make_unique<OperatorNode>(
            binary_ops[op_dist(gen)],
            std::move(children),
            ops
        );
    }

    // Helper to create guaranteed full binary tree of exact depth
    static std::unique_ptr<GPNode> createFullBinaryTree(
        int exact_depth,
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        if (exact_depth <= 1) {
            return createLeafNode(config, gen);
        }

        static const std::vector<std::string> binary_ops = {"add", "mul", "sub", "div"};
        std::uniform_int_distribution<size_t> op_dist(0, binary_ops.size() - 1);
        
        std::vector<std::unique_ptr<GPNode>> children;
        children.push_back(createFullBinaryTree(exact_depth - 1, config, ops, gen));
        children.push_back(createFullBinaryTree(exact_depth - 1, config, ops, gen));
        
        return std::make_unique<OperatorNode>(
            binary_ops[op_dist(gen)],
            std::move(children),
            ops
        );
    }

    static std::unique_ptr<GPNode> multiPointCrossover(
        const GPNode* parent1,
        const GPNode* parent2,
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        if (!parent1 || !parent2) {
            return createFullBinaryTree(config.min_tree_depth, config, ops, gen);
        }

        const int MAX_ATTEMPTS = 5;
        
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            try {
                auto offspring = parent1->clone();
                if (!offspring) continue;
                
                std::vector<std::tuple<const GPNode*, int, bool>> valid_nodes1;
                std::vector<std::tuple<const GPNode*, int, bool>> valid_nodes2;
                
                // Modified to ensure minimum depth
                auto collectValidNodes = [&config](const GPNode* root, 
                    std::vector<std::tuple<const GPNode*, int, bool>>& nodes) {
                    std::function<void(const GPNode*, int)> traverse = 
                        [&](const GPNode* node, int depth) {
                        if (!node) return;
                        
                        if (const auto* op_node = dynamic_cast<const OperatorNode*>(node)) {
                            const auto& children = op_node->getChildren();
                            
                            // Check if this subtree maintains minimum depth
                            bool maintains_min_depth = hasMinimumDepth(node, config.min_tree_depth - depth);
                            
                            if (maintains_min_depth && depth + node->depth() < config.max_tree_depth) {
                                nodes.emplace_back(node, depth, true);
                            }
                            
                            for (const auto& child : children) {
                                if (child) traverse(child.get(), depth + 1);
                            }
                        }
                    };
                    
                    traverse(root, 0);
                };

                collectValidNodes(offspring.get(), valid_nodes1);
                collectValidNodes(parent2, valid_nodes2);

                if (valid_nodes1.empty() || valid_nodes2.empty()) {
                    return createFullBinaryTree(config.min_tree_depth, config, ops, gen);
                }

                // Select random valid nodes
                std::uniform_int_distribution<size_t> dist1(0, valid_nodes1.size() - 1);
                std::uniform_int_distribution<size_t> dist2(0, valid_nodes2.size() - 1);
                
                auto [target_node, target_depth, _1] = valid_nodes1[dist1(gen)];
                auto [source_node, source_depth, _2] = valid_nodes2[dist2(gen)];

                // Clone and validate subtree
                auto new_subtree = source_node->clone();
                if (!new_subtree || !hasMinimumDepth(new_subtree.get(), config.min_tree_depth - target_depth)) {
                    continue;
                }

                // Test replacement
                auto test_tree = offspring->clone();
                if (!test_tree) continue;

                try {
                    // Find corresponding node
                    std::function<const GPNode*(const GPNode*, const GPNode*)> findCorresponding;
                    findCorresponding = [&findCorresponding](const GPNode* root, const GPNode* target) 
                        -> const GPNode* {
                        if (!root || !target) return nullptr;
                        if (root == target) return root;
                        
                        if (const auto* op_node = dynamic_cast<const OperatorNode*>(root)) {
                            for (const auto& child : op_node->getChildren()) {
                                if (auto* found = findCorresponding(child.get(), target)) {
                                    return found;
                                }
                            }
                        }
                        return nullptr;
                    };

                    auto* test_target = findCorresponding(test_tree.get(), target_node);
                    if (!test_target) continue;

                    // Test replacement
                    test_tree->replaceSubtree(const_cast<GPNode*>(test_target), new_subtree->clone());
                    
                    // Validate minimum depth
                    if (!hasMinimumDepth(test_tree.get(), config.min_tree_depth)) {
                        continue;
                    }

                    // Perform actual replacement
                    offspring->replaceSubtree(const_cast<GPNode*>(target_node), std::move(new_subtree));
                    
                    // Final validation
                    if (hasMinimumDepth(offspring.get(), config.min_tree_depth)) {
                        return offspring;
                    }
                } catch (...) {
                    continue;
                }
            } catch (...) {
                continue;
            }
        }

        // If all attempts fail, create guaranteed minimum depth tree
        return createFullBinaryTree(config.min_tree_depth, config, ops, gen);
    }

    // Create a deeper initial tree
    static std::unique_ptr<GPNode> createDeepTree(
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        // Target a depth closer to max_depth while ensuring minimum
        int target_depth = std::max(
            config.min_tree_depth,
            config.min_tree_depth + (config.max_tree_depth - config.min_tree_depth) * 3 / 4
        );

        std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
        static const std::vector<std::string> ops_list = {"add", "mul", "sub", "div"};
        std::uniform_int_distribution<size_t> op_dist(0, ops_list.size() - 1);
        
        std::function<std::unique_ptr<GPNode>(int)> buildDeep = 
            [&](int remaining_depth) -> std::unique_ptr<GPNode> {
            if (remaining_depth <= 1) {
                return std::make_unique<FeatureNode>(feature_dist(gen));
            }
            
            // Higher probability of creating operator nodes at lower depths
            float op_prob = std::min(0.9f, 0.5f + (remaining_depth / static_cast<float>(target_depth)));
            std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
            
            if (prob_dist(gen) < op_prob) {
                std::vector<std::unique_ptr<GPNode>> children;
                children.push_back(buildDeep(remaining_depth - 1));
                children.push_back(buildDeep(remaining_depth - 1));
                return std::make_unique<OperatorNode>(ops_list[op_dist(gen)], std::move(children), ops);
            } else {
                return std::make_unique<FeatureNode>(feature_dist(gen));
            }
        };

        return buildDeep(target_depth);
    }

    // Enhanced mutation to encourage deeper trees
    static void mutateNode(
        GPNode* node,
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen,
        int current_depth = 0
    ) {
        if (!node) return;

        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        if (auto* op_node = dynamic_cast<OperatorNode*>(node)) {
            // Higher chance of mutation at deeper levels to encourage complexity
            float mutation_prob = std::min(0.8f, 0.3f + (current_depth / static_cast<float>(config.max_tree_depth)));
            
            if (prob_dist(gen) < mutation_prob) {
                // Chance to add more complexity by replacing a leaf with an operator
                for (auto& child : op_node->getMutableChildren()) {
                    if (child && child->isLeaf() && 
                        current_depth + 2 < config.max_tree_depth && 
                        prob_dist(gen) < 0.4f) {
                        // Create a new operator node
                        std::vector<std::unique_ptr<GPNode>> new_children;
                        std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
                        
                        new_children.push_back(std::make_unique<FeatureNode>(feature_dist(gen)));
                        new_children.push_back(std::make_unique<FeatureNode>(feature_dist(gen)));
                        
                        static const std::vector<std::string> ops_list = {"add", "mul", "sub", "div"};
                        std::uniform_int_distribution<size_t> op_dist(0, ops_list.size() - 1);
                        
                        child = std::make_unique<OperatorNode>(
                            ops_list[op_dist(gen)], 
                            std::move(new_children), 
                            ops
                        );
                    }
                }
            }
            
            // Recurse on children
            for (auto& child : op_node->getMutableChildren()) {
                if (child) {
                    mutateNode(child.get(), config, ops, gen, current_depth + 1);
                }
            }
        } else if (auto* feature_node = dynamic_cast<FeatureNode*>(node)) {
            // Higher chance to replace feature nodes with operators at shallow depths
            float replace_prob = std::max(0.0f, 0.6f - (current_depth / static_cast<float>(config.max_tree_depth)));
            
            if (current_depth + 2 < config.max_tree_depth && prob_dist(gen) < replace_prob) {
                // Create a new operator node to replace this feature node
                std::vector<std::unique_ptr<GPNode>> new_children;
                std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
                
                new_children.push_back(std::make_unique<FeatureNode>(feature_dist(gen)));
                new_children.push_back(std::make_unique<FeatureNode>(feature_dist(gen)));
                
                static const std::vector<std::string> ops_list = {"add", "mul", "sub", "div"};
                std::uniform_int_distribution<size_t> op_dist(0, ops_list.size() - 1);
                
                *node = OperatorNode(ops_list[op_dist(gen)], std::move(new_children), ops);
            }
        }
    }
        
    // Create exact depth tree - no changes needed here
    static std::unique_ptr<GPNode> createExactDepthTree(
        int exact_depth,
        const GPConfig& config,
        std::shared_ptr<GPOperations> ops,
        std::mt19937& gen
    ) {
        std::uniform_int_distribution<int> feature_dist(0, config.n_features - 1);
        
        if (exact_depth == 1) {
            return std::make_unique<FeatureNode>(feature_dist(gen));
        }
        
        std::vector<std::unique_ptr<GPNode>> children;
        children.push_back(createExactDepthTree(exact_depth - 1, config, ops, gen));
        children.push_back(createExactDepthTree(exact_depth - 1, config, ops, gen));
        
        static const std::vector<std::string> ops_list = {"add", "mul", "sub"};
        std::uniform_int_distribution<size_t> op_dist(0, ops_list.size() - 1);
        return std::make_unique<OperatorNode>(ops_list[op_dist(gen)], std::move(children), ops);
    }

    // Modified to accept const pointers
    static bool isFullBinaryToDepth(const GPNode* node, int current_depth, int required_depth) {
        if (!node) return false;
        
        if (current_depth >= required_depth) return true;
        
        const auto* op_node = dynamic_cast<const OperatorNode*>(node);
        if (!op_node) return false;
        
        const auto& children = op_node->getChildren();
        if (children.size() != 2) return false;
        
        return isFullBinaryToDepth(children[0].get(), current_depth + 1, required_depth) &&
            isFullBinaryToDepth(children[1].get(), current_depth + 1, required_depth);
    }
};

// Initialize the thread_local operation counter
thread_local int TreeOperations::operation_counter = 0;

// Individual class representing a collection of trees
// Function to combine two hash values
inline size_t hashCombine(size_t seed, size_t hash) {
    return seed ^ (hash + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

class EvaluationState {
private:
    static std::mutex global_mutex;
    static std::atomic<int> active_evaluations;  // No initialization here
    static std::atomic<bool> is_evaluating;      // No initialization here
    static thread_local std::vector<float> result;
    bool has_lock{false};
    
public:
    EvaluationState() {
        std::lock_guard<std::mutex> lock(global_mutex);
        if (!is_evaluating.load()) {
            is_evaluating.store(true);
            has_lock = true;
            active_evaluations.fetch_add(1);
        }
    }
    
    ~EvaluationState() {
        if (has_lock) {
            std::lock_guard<std::mutex> lock(global_mutex);
            if (active_evaluations.fetch_sub(1) == 1) {
                is_evaluating.store(false);
            }
        }
    }
    
    bool hasLock() const { return has_lock; }
    
    static const std::vector<float>& getResult() {
        if (result.empty()) {
            result.assign(1, 0.0f);
        }
        return result;
    }
    
    static void setResult(float value) {
        result.assign(1, value);
    }
};

// Define and initialize static members
std::mutex EvaluationState::global_mutex;
std::atomic<int> EvaluationState::active_evaluations(0);
std::atomic<bool> EvaluationState::is_evaluating(false);
thread_local std::vector<float> EvaluationState::result{0.0f};

class Individual {
private:
    mutable std::mutex trees_mutex;  // Reader-writer lock for trees
    std::vector<std::unique_ptr<GPNode>> trees;
    float fitness;
    std::shared_ptr<GPOperations> ops;
    const GPConfig& config;  // Reference to config
    mutable std::mutex eval_mutex;
    mutable std::atomic<int> eval_count{0};
    mutable std::atomic<bool> evaluation_in_progress{false};
    static constexpr int MAX_RETRIES = 3;

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
                hash = hashCombine(hash, 0x1234567); // Magic number for feature node 
                hash = hashCombine(hash, feature_node->getFeatureIndex());
            }

            // Hash node depth
            hash = hashCombine(hash, node->depth());
        }

        return hash;
    }

public:
    Individual(std::vector<std::unique_ptr<GPNode>> t, 
              std::shared_ptr<GPOperations> operations,
              const GPConfig& cfg)
        : trees(std::move(t))
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

    // Copy constructor with proper mutex handling
    Individual(const Individual& other)
        : fitness(other.getFitness())
        , ops(other.ops)
        , config(other.config) 
    {
        std::lock_guard<std::mutex> other_lock(other.trees_mutex);
        trees.reserve(other.trees.size());
        for (const auto& tree : other.trees) {
            if (tree) {
                trees.push_back(tree->clone());
            }
        }
    }

    // Move constructor
    Individual(Individual&& other) noexcept
        : trees(std::move(other.trees))
        , fitness(other.fitness)
        , ops(std::move(other.ops))
        , config(other.config)
    {}

    // Copy assignment
    Individual& operator=(Individual other) {
        std::lock(trees_mutex, other.trees_mutex);
        std::lock_guard<std::mutex> this_lock(trees_mutex, std::adopt_lock);
        std::lock_guard<std::mutex> other_lock(other.trees_mutex, std::adopt_lock);
        
        std::swap(trees, other.trees);
        fitness = other.getFitness();
        ops = other.ops;
        return *this;
    }

    // Reset all trees
    void resetTrees() {
        std::lock_guard<std::mutex> lock(eval_mutex);
        trees.clear();
    }

    // Compute hash value for the individual
    size_t computeHash() const {
        size_t hash = 0;
        for (const auto& tree : trees) {
            hash = hashCombine(hash, computeTreeHash(tree));
        }
        return hash;
    }

    // In Individual class:
    std::vector<float> evaluate(const std::vector<float>& input) const {
        std::lock_guard<std::mutex> lock(trees_mutex);
    
        try {
            if (input.empty() || input.size() != config.n_features) {
                return std::vector<float>{0.0f};
            }

            std::vector<float> result;
            result.reserve(trees.size());
            
            for (const auto& tree : trees) {
                if (!tree) continue;
                
                try {
                    auto tree_output = tree->evaluate(input);
                    if (!tree_output.empty() && std::isfinite(tree_output[0])) {
                        result.push_back(tree_output[0]);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Tree evaluation error: " << e.what() << std::endl;
                    continue;
                }
            }

            return result.empty() ? std::vector<float>{0.0f} : result;
            
        } catch (const std::exception& e) {
            std::cerr << "Individual evaluation error: " << e.what() << std::endl;
            return std::vector<float>{0.0f};
        }
    }

    void setFitness(float f) { 
        fitness = f; 
    }
    
    float getFitness() const { 
        return fitness; 
    }

    void mutate(std::mt19937& gen) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& tree : trees) {
            if (tree && dist(gen) < config.mutation_prob) {
                TreeOperations::mutateNode(tree.get(), config, ops, gen);
            }
        }
    }


    int totalSize() const {
        int size = 0;
        for (const auto& tree : trees) {
            if (tree) {
                size += tree->size();
            }
        }
        return size;
    }

    const std::unique_ptr<GPNode>& getTree(int index) const {
        return trees[index];
    }

    bool validateDepth() const {
        for (const auto& tree : trees) {
            if (!tree) {
                return false;
            }
            int depth = tree->depth();
            if (depth < config.min_tree_depth || depth > config.max_tree_depth) {
                return false;
            }
        }
        return true;
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

struct ThresholdStats {
    std::deque<float> recent_distances;  // Stores recent distances
    float current_threshold;
    float positive_mean;
    float negative_mean;
    mutable std::mutex stats_mutex;  // Make mutex mutable so it can be locked in const functions

    ThresholdStats(float initial_threshold) 
        : current_threshold(initial_threshold)
        , positive_mean(initial_threshold)
        , negative_mean(initial_threshold * 2.0f)
    {}

    void updateStats(float distance, bool is_same_class) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        recent_distances.push_back(distance);
        
        // Keep only recent distances
        while (recent_distances.size() > 1000) {
            recent_distances.pop_front();
        }

        // Recalculate means
        std::vector<float> pos_distances;
        std::vector<float> neg_distances;
        
        size_t window_start = recent_distances.size() > 100 ? 
                             recent_distances.size() - 100 : 0;
        
        for (size_t i = window_start; i < recent_distances.size(); ++i) {
            if (is_same_class) {
                pos_distances.push_back(recent_distances[i]);
            } else {
                neg_distances.push_back(recent_distances[i]);
            }
        }

        if (!pos_distances.empty()) {
            positive_mean = std::accumulate(pos_distances.begin(), 
                                         pos_distances.end(), 0.0f) / pos_distances.size();
        }
        if (!neg_distances.empty()) {
            negative_mean = std::accumulate(neg_distances.begin(), 
                                         neg_distances.end(), 0.0f) / neg_distances.size();
        }
    }

    float getAdaptiveThreshold() const {
        std::lock_guard<std::mutex> lock(stats_mutex);  // Now works with mutable mutex
        return (positive_mean + negative_mean) / 2.0f;
    }
};

class ProgressBar {
private:
    size_t total;
    size_t current;
    size_t bar_width;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update;
    std::string description;
    
public:
    ProgressBar(size_t total_, size_t width = 50, const std::string& desc = "")
        : total(total_)
        , current(0)
        , bar_width(width)
        , start_time(std::chrono::steady_clock::now())
        , last_update(start_time)
        , description(desc)
    {
        // Print initial bar
        printBar();
    }
    
    void update(size_t n = 1) {
        current += n;
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
        
        // Update at most once every 100ms to avoid too frequent refreshes
        if (duration > 100 || current >= total) {
            printBar();
            last_update = now;
        }
    }
    
    void printBar() {
        float progress = static_cast<float>(current) / total;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        
        // Calculate ETA
        float speed = elapsed > 0 ? static_cast<float>(current) / elapsed : 0;
        int eta = speed > 0 ? (total - current) / speed : 0;
        
        // Create the progress bar
        int pos = static_cast<int>(bar_width * progress);
        
        std::cout << "\r" << description << " [";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << static_cast<int>(progress * 100.0f) << "% "
                 << current << "/" << total << " "
                 << "[" << elapsed << "s elapsed, ETA " << eta << "s]" << std::flush;
        
        if (current >= total) {
            std::cout << std::endl;
        }
    }
};

class BinaryClassifier {
private:
    std::vector<std::unique_ptr<GPNode>> trees;
    const GPConfig& config;
    std::shared_ptr<GPOperations> ops;
    mutable std::mutex predict_mutex;

    struct TrainingStats {
        float accuracy = 0.0f;
        float sensitivity = 0.0f;
        float specificity = 0.0f;
        int true_positives = 0;
        int true_negatives = 0;
        int false_positives = 0;
        int false_negatives = 0;

        void print() const {
            std::cout << "Training Statistics:\n"
                     << "  Accuracy: " << (accuracy * 100.0f) << "%\n"
                     << "  Sensitivity: " << (sensitivity * 100.0f) << "%\n"
                     << "  Specificity: " << (specificity * 100.0f) << "%\n"
                     << "  True Positives: " << true_positives << "\n"
                     << "  True Negatives: " << true_negatives << "\n"
                     << "  False Positives: " << false_positives << "\n"
                     << "  False Negatives: " << false_negatives << std::endl;
        }
    };

    std::vector<const DataPoint*> createBalancedBatch(
        const std::vector<DataPoint>& data, 
        size_t batch_size
    ) const {
        std::vector<const DataPoint*> positive_samples;
        std::vector<const DataPoint*> negative_samples;
        
        for (const auto& point : data) {
            if (point.label > 0.5f) {
                positive_samples.push_back(&point);
            } else {
                negative_samples.push_back(&point);
            }
        }

        // Determine number of samples from each class
        size_t pos_samples = std::min(batch_size / 2, positive_samples.size());
        size_t neg_samples = std::min(batch_size - pos_samples, negative_samples.size());
        
        // Readjust positive samples if we couldn't get enough negative ones
        pos_samples = std::min(batch_size - neg_samples, positive_samples.size());

        // Create batch
        std::vector<const DataPoint*> batch;
        batch.reserve(pos_samples + neg_samples);

        // Shuffle and select samples
        std::shuffle(positive_samples.begin(), positive_samples.end(), getThreadLocalRNG());
        std::shuffle(negative_samples.begin(), negative_samples.end(), getThreadLocalRNG());

        batch.insert(batch.end(), positive_samples.begin(), positive_samples.begin() + pos_samples);
        batch.insert(batch.end(), negative_samples.begin(), negative_samples.begin() + neg_samples);
        
        // Final shuffle
        std::shuffle(batch.begin(), batch.end(), getThreadLocalRNG());
        
        return batch;
    }

    // Evaluate a batch of samples
    TrainingStats evaluateBatch(const std::vector<const DataPoint*>& batch) const {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        
        #pragma omp parallel for reduction(+:tp,tn,fp,fn)
        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& point = *batch[i];
            float prediction = predict(point.anchor, point.compare);
            bool predicted_same = prediction >= config.decision_threshold;
            bool actually_same = point.label > 0.5f;

            if (actually_same) {
                if (predicted_same) ++tp;
                else ++fn;
            } else {
                if (predicted_same) ++fp;
                else ++tn;
            }
        }

        // Create stats object and fill it with results
        TrainingStats stats;
        stats.true_positives = tp;
        stats.true_negatives = tn;
        stats.false_positives = fp;
        stats.false_negatives = fn;

        // Calculate metrics
        float total_positive = stats.true_positives + stats.false_negatives;
        float total_negative = stats.true_negatives + stats.false_positives;
        float total = total_positive + total_negative;

        stats.sensitivity = total_positive > 0 ? 
            static_cast<float>(stats.true_positives) / total_positive : 0.0f;
        stats.specificity = total_negative > 0 ? 
            static_cast<float>(stats.true_negatives) / total_negative : 0.0f;
        stats.accuracy = total > 0 ? 
            static_cast<float>(stats.true_positives + stats.true_negatives) / total : 0.0f;

        return stats;
    }

    std::vector<std::unique_ptr<GPNode>> cloneTrees() const {
        std::vector<std::unique_ptr<GPNode>> cloned;
        cloned.reserve(trees.size());
        for (const auto& tree : trees) {
            if (tree) {
                cloned.push_back(tree->clone());
            }
        }
        return cloned;
    }
    
    // Helper function to apply sigmoid
    float sigmoid(float x) const {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // Helper to normalize feature vector
    std::vector<float> normalizeFeatures(const std::vector<float>& features) const {
        if (features.empty()) return {};
        
        std::vector<float> normalized(features.size());
        float mean = 0.0f;
        float var = 0.0f;
        
        // Calculate mean
        for (float f : features) {
            mean += f;
        }
        mean /= features.size();
        
        // Calculate variance
        for (float f : features) {
            float diff = f - mean;
            var += diff * diff;
        }
        var = std::sqrt(var / features.size() + 1e-6f);
        
        // Normalize
        for (size_t i = 0; i < features.size(); ++i) {
            normalized[i] = (features[i] - mean) / (var + 1e-6f);
        }
        
        return normalized;
    }

    // Helper to combine features with various operations
    std::vector<float> combineFeatures(
        const std::vector<float>& anchor, 
        const std::vector<float>& compare
    ) const {
        if (anchor.empty() || compare.empty() || anchor.size() != compare.size()) {
            return {};
        }
        
        // Output will contain: [anchor, compare, diff, product, max, min]
        std::vector<float> combined;
        combined.reserve(anchor.size() * 6);
        
        // Add original features
        combined.insert(combined.end(), anchor.begin(), anchor.end());
        combined.insert(combined.end(), compare.begin(), compare.end());
        
        // Add difference features
        for (size_t i = 0; i < anchor.size(); ++i) {
            combined.push_back(anchor[i] - compare[i]);
        }
        
        // Add product features
        for (size_t i = 0; i < anchor.size(); ++i) {
            combined.push_back(anchor[i] * compare[i]);
        }
        
        // Add max features
        for (size_t i = 0; i < anchor.size(); ++i) {
            combined.push_back(std::max(anchor[i], compare[i]));
        }
        
        // Add min features
        for (size_t i = 0; i < anchor.size(); ++i) {
            combined.push_back(std::min(anchor[i], compare[i]));
        }
        
        return normalizeFeatures(combined);
    }

public:
    BinaryClassifier(const GPConfig& cfg, std::shared_ptr<GPOperations> operations) 
        : config(cfg)
        , ops(operations) 
    {
        initializeTrees();
    }
    
    // Initialize the classifier trees
    void initializeTrees() {
        trees.clear();
        trees.reserve(config.classifier_trees);
        
        for (int i = 0; i < config.classifier_trees; ++i) {
            auto tree = createRandomTree(config.min_tree_depth, config.max_tree_depth);
            if (tree) {
                trees.push_back(std::move(tree));
            }
        }
    }
    
    // Create a random tree with specific depth constraints
    std::unique_ptr<GPNode> createRandomTree(int min_depth, int max_depth) {
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> feature_dist(0, config.n_features * 6 - 1);  // 6x features
        
        // Helper function to create tree recursively
        std::function<std::unique_ptr<GPNode>(int, int)> buildTree = 
            [&](int depth, int max_allowed) -> std::unique_ptr<GPNode> {
            
            if (depth >= max_allowed) {
                return std::make_unique<FeatureNode>(feature_dist(getThreadLocalRNG()));
            }
            
            if (depth >= min_depth && prob_dist(getThreadLocalRNG()) < 0.3f) {
                return std::make_unique<FeatureNode>(feature_dist(getThreadLocalRNG()));
            }
            
            // Create operator node
            std::vector<std::string> possible_ops = {"add", "sub", "mul", "div"};
            std::uniform_int_distribution<size_t> op_dist(0, possible_ops.size() - 1);
            
            std::vector<std::unique_ptr<GPNode>> children;
            children.push_back(buildTree(depth + 1, max_allowed));
            children.push_back(buildTree(depth + 1, max_allowed));
            
            return std::make_unique<OperatorNode>(
                possible_ops[op_dist(getThreadLocalRNG())],
                std::move(children),
                ops
            );
        };
        
        return buildTree(0, max_depth);
    }
    
    // Main prediction function
    float predict(const std::vector<float>& anchor, const std::vector<float>& compare) const {
        std::lock_guard<std::mutex> lock(predict_mutex);
        
        if (anchor.empty() || compare.empty() || anchor.size() != compare.size()) {
            return 0.0f;
        }
        
        // Combine features with various operations
        std::vector<float> combined = combineFeatures(anchor, compare);
        if (combined.empty()) return 0.0f;
        
        // Get predictions from each tree
        std::vector<float> predictions;
        predictions.reserve(trees.size());
        
        for (const auto& tree : trees) {
            if (!tree) continue;
            
            try {
                auto output = tree->evaluate(combined);
                if (!output.empty() && std::isfinite(output[0])) {
                    predictions.push_back(sigmoid(output[0] * config.feature_scale));
                }
            } catch (...) {
                continue;
            }
        }
        
        if (predictions.empty()) return 0.0f;
        
        // Sort predictions and remove outliers
        std::sort(predictions.begin(), predictions.end());
        if (predictions.size() > 3) {
            // Remove most extreme predictions
            predictions.erase(predictions.begin());
            predictions.pop_back();
        }
        
        // Count strong votes
        int positive_votes = 0;
        int negative_votes = 0;
        float total_confidence = 0.0f;
        float weighted_sum = 0.0f;
        
        for (float pred : predictions) {
            float confidence = std::abs(pred - 0.5f) * 2.0f;  // Scale to [0,1]
            total_confidence += confidence;
            weighted_sum += pred * confidence;
            
            if (pred >= 0.5f) {
                positive_votes++;
            } else {
                negative_votes++;
            }
        }
        
        // Check if we have enough confident votes
        if (std::max(positive_votes, negative_votes) < config.min_votes) {
            return 0.5f;  // Not confident enough
        }
        
        // Return confidence-weighted average
        return total_confidence > 0.0f ? weighted_sum / total_confidence : 0.5f;
    }
    
    // Function to mutate trees
    void mutate() {
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        for (auto& tree : trees) {
            if (!tree) continue;
            
            if (prob_dist(getThreadLocalRNG()) < config.mutation_prob) {
                tree->mutate(getThreadLocalRNG(), config);
            }
        }
    }
    
    // Function to perform crossover with another classifier
    void crossover(const BinaryClassifier& other) {
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < trees.size() && i < other.trees.size(); ++i) {
            if (prob_dist(getThreadLocalRNG()) < config.crossover_prob) {
                if (trees[i] && other.trees[i]) {
                    auto new_tree = TreeOperations::multiPointCrossover(
                        trees[i].get(), 
                        other.trees[i].get(), 
                        config, 
                        ops, 
                        getThreadLocalRNG()
                    );
                    if (new_tree) {
                        trees[i] = std::move(new_tree);
                    }
                }
            }
        }
    }
    
    // Function to get classifier size (total nodes across all trees)
    int getSize() const {
        int total = 0;
        for (const auto& tree : trees) {
            if (tree) {
                total += tree->size();
            }
        }
        return total;
    }
    
    // Function to clone the classifier
    std::unique_ptr<BinaryClassifier> clone() const {
        auto new_classifier = std::make_unique<BinaryClassifier>(config, ops);
        new_classifier->trees.clear();
        
        for (const auto& tree : trees) {
            if (tree) {
                new_classifier->trees.push_back(tree->clone());
            }
        }
        
        return new_classifier;
    }

    // Main training method
    void train(const std::vector<DataPoint>& trainData, 
              const std::vector<DataPoint>& valData,
              int max_epochs = 100,
              size_t batch_size = 64) {
        if (trainData.empty() || valData.empty()) {
            std::cerr << "Empty training or validation data" << std::endl;
            return;
        }

        // Population of trees
        const size_t pop_size = config.population_size;  // Moved up
        const size_t elite_size = config.elite_size;     // Moved up
        std::vector<std::vector<std::unique_ptr<GPNode>>> population;

        // Initialize population
        population.reserve(pop_size);
        for (size_t i = 0; i < pop_size; ++i) {
            std::vector<std::unique_ptr<GPNode>> new_trees;
            new_trees.reserve(config.classifier_trees);
            for (int j = 0; j < config.classifier_trees; ++j) {
                new_trees.push_back(createRandomTree(config.min_tree_depth, config.max_tree_depth));
            }
            population.push_back(std::move(new_trees));
        }

        float best_val_accuracy = 0.0f;
        float best_train_accuracy = 0.0f;
        int epochs_without_improvement = 0;
        auto best_trees = cloneTrees();

        // Create fixed validation set
        auto validation_batch = createBalancedBatch(valData, std::min(valData.size(), size_t(1000)));

        ProgressBar progress(max_epochs, 50, "Training Progress");

        // Training loop
        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            // Keep dropout enabled for training fitness evaluation
            std::vector<std::pair<float, size_t>> fitness_scores;
            fitness_scores.reserve(pop_size);

            for (size_t i = 0; i < population.size(); ++i) {
                auto backup_trees = cloneTrees();
                trees = std::move(population[i]);

                auto batch = createBalancedBatch(trainData, batch_size);
                auto stats = evaluateBatch(batch);  // With dropout
                float fitness = (stats.sensitivity + stats.specificity) / 2.0f;

                fitness_scores.emplace_back(fitness, i);

                population[i] = cloneTrees();
                trees = std::move(backup_trees);
            }

            // Sort by fitness (using dropout-enabled scores)
            std::sort(fitness_scores.begin(), fitness_scores.end(),
                     std::greater<std::pair<float, size_t>>());

            // Create new population starting with elites
            std::vector<std::vector<std::unique_ptr<GPNode>>> new_population;
            new_population.reserve(pop_size);

            // Always preserve current best trees
            new_population.push_back(cloneTreeVector(population[fitness_scores[0].second]));

            // Add remaining elites
            for (size_t i = 0; i < elite_size - 1 && i < fitness_scores.size(); ++i) {
                new_population.push_back(cloneTreeVector(population[fitness_scores[i].second]));
            }

            // Fill rest of population through tournament selection and crossover
            while (new_population.size() < pop_size) {
                size_t parent1_idx = tournamentSelect(fitness_scores);
                size_t parent2_idx = tournamentSelect(fitness_scores);

                auto offspring = createOffspring(population[parent1_idx], population[parent2_idx]);

                if (std::uniform_real_distribution<float>(0.0f, 1.0f)(getThreadLocalRNG()) < 
                    config.mutation_prob) {
                    mutateTreeVector(offspring);
                }

                new_population.push_back(std::move(offspring));
            }

            // Only disable dropout for final validation metrics
            trees = cloneTreeVector(population[fitness_scores[0].second]);
            
            // Get training accuracy with dropout (real training conditions)
            auto train_stats = evaluateBatch(createBalancedBatch(trainData, 1000));
            float train_accuracy = (train_stats.sensitivity + train_stats.specificity) / 2.0f;

            // Get validation accuracy without dropout
            ops->setTraining(false);  // Disable dropout only for validation
            auto val_stats = evaluateBatch(validation_batch);
            float val_accuracy = (val_stats.sensitivity + val_stats.specificity) / 2.0f;
            ops->setTraining(true);   // Re-enable dropout

            // Track best model based on validation accuracy
            if (val_accuracy > best_val_accuracy) {
                best_val_accuracy = val_accuracy;
                best_train_accuracy = train_accuracy;
                best_trees = cloneTrees();
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement++;
            }

            // Print progress
            std::cout << "\nEpoch " << epoch + 1
                     << "\nTraining Accuracy (with dropout): " << train_accuracy * 100.0f << "%"
                     << "\nValidation Accuracy: " << val_accuracy * 100.0f << "%"
                     << "\nBest Training Accuracy: " << best_train_accuracy * 100.0f << "%"
                     << "\nBest Validation Accuracy: " << best_val_accuracy * 100.0f << "%"
                     << "\nEpochs without improvement: " << epochs_without_improvement << std::endl;

            // Early stopping check
            if (epochs_without_improvement >= 10) {
                std::cout << "Early stopping triggered" << std::endl;
                // break;
            }

            // Replace population
            population = std::move(new_population);
            progress.update();
        }

        // Final evaluation with best model and no dropout
        ops->setTraining(false);
        trees = std::move(best_trees);
        
        // Final validation score
        auto final_val_stats = evaluateBatch(validation_batch);
        std::cout << "\nFinal Model Performance (No Dropout):"
                  << "\nValidation Accuracy: " << (final_val_stats.accuracy * 100.0f) << "%"
                  << "\nSensitivity: " << (final_val_stats.sensitivity * 100.0f) << "%"
                  << "\nSpecificity: " << (final_val_stats.specificity * 100.0f) << "%" << std::endl;
    }

private:
    // Helper methods for population management
    std::vector<std::unique_ptr<GPNode>> cloneTreeVector(
        const std::vector<std::unique_ptr<GPNode>>& tree_vec) const {
        std::vector<std::unique_ptr<GPNode>> cloned;
        cloned.reserve(tree_vec.size());
        for (const auto& tree : tree_vec) {
            if (tree) {
                cloned.push_back(tree->clone());
            }
        }
        return cloned;
    }

    void mutateTreeVector(std::vector<std::unique_ptr<GPNode>>& tree_vec) {
        for (auto& tree : tree_vec) {
            if (tree && std::uniform_real_distribution<float>(0.0f, 1.0f)(getThreadLocalRNG()) < 
                config.mutation_prob) {
                tree->mutate(getThreadLocalRNG(), config);
            }
        }
    }

    size_t tournamentSelect(const std::vector<std::pair<float, size_t>>& fitness_scores) {
        std::uniform_int_distribution<size_t> dist(0, fitness_scores.size() - 1);
        size_t best_idx = dist(getThreadLocalRNG());
        float best_fitness = fitness_scores[best_idx].first;

        for (int i = 1; i < config.tournament_size; ++i) {
            size_t idx = dist(getThreadLocalRNG());
            if (fitness_scores[idx].first > best_fitness) {
                best_idx = idx;
                best_fitness = fitness_scores[idx].first;
            }
        }

        return fitness_scores[best_idx].second;
    }

    std::vector<std::unique_ptr<GPNode>> createOffspring(
        const std::vector<std::unique_ptr<GPNode>>& parent1,
        const std::vector<std::unique_ptr<GPNode>>& parent2) {
        std::vector<std::unique_ptr<GPNode>> offspring;
        offspring.reserve(parent1.size());

        for (size_t i = 0; i < parent1.size() && i < parent2.size(); ++i) {
            if (std::uniform_real_distribution<float>(0.0f, 1.0f)(getThreadLocalRNG()) < 
                config.crossover_prob) {
                auto new_tree = TreeOperations::multiPointCrossover(
                    parent1[i].get(), parent2[i].get(), config, ops, getThreadLocalRNG());
                offspring.push_back(std::move(new_tree));
            } else {
                offspring.push_back(parent1[i]->clone());
            }
        }

        return offspring;
    }
};

class ContrastiveGP {
private:
    GPConfig config;
    std::shared_ptr<GPOperations> ops;
    std::vector<Individual> population;
    static thread_local std::mt19937 rng;
    FitnessCache fitness_cache; // Add as class member
    std::mutex cache_mutex; // Declare cache_mutex
    ThresholdStats threshold_stats;

    std::mt19937& getGen() {
        return getThreadLocalRNG();  // Use the global initialization function
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

        // Calculate Euclidean distance component
        float euclidean_dist = 0.0f;
        for (size_t i = 0; i < v1.size(); ++i) {
            float diff = v1[i] - v2[i];
            euclidean_dist += diff * diff;
        }
        euclidean_dist = std::sqrt(euclidean_dist);

        // Calculate normalized cosine similarity component
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        float dot_product = 0.0f;
        
        for (size_t i = 0; i < v1.size(); ++i) {
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
            dot_product += v1[i] * v2[i];
        }

        // Add small epsilon to prevent division by zero
        constexpr float epsilon = 1e-10f;
        norm1 = std::sqrt(norm1 + epsilon);
        norm2 = std::sqrt(norm2 + epsilon);

        float cosine_similarity = dot_product / (norm1 * norm2);
        
        // Clamp similarity to [-1, 1] range
        cosine_similarity = std::max(-1.0f, std::min(1.0f, cosine_similarity));
        
        // Convert similarity to distance (1 - similarity) and scale it
        float cosine_distance = (1.0f - cosine_similarity);

        // Combine distances with exponential scaling to increase separation
        float combined_distance = 0.5f * euclidean_dist + 0.5f * std::exp(cosine_distance) - 1.0f;
        
        // Apply sigmoid transformation to squash extreme values and increase separation
        float scaled_distance = 1.0f / (1.0f + std::exp(-2.0f * (combined_distance - 0.5f)));

        // Check for NaN or inf
        if (std::isnan(scaled_distance) || std::isinf(scaled_distance)) {
            std::cerr << "Warning: Invalid distance calculated" << std::endl;
            return std::numeric_limits<float>::max();
        }

        return scaled_distance;
    }

    float calculateAccuracy(const Individual& ind, const std::vector<DataPoint>& data) {
        if (data.empty()) {
            std::cerr << "Empty dataset provided" << std::endl;
            return 0.0f;
        }

        std::atomic<int> true_positives{0};
        std::atomic<int> true_negatives{0};
        std::atomic<int> total_positives{0};
        std::atomic<int> total_negatives{0};
        
        const size_t SAFE_BATCH_SIZE = 32;
        std::atomic<size_t> processed{0};
        
        std::vector<std::future<void>> futures;
        const size_t num_threads = config.num_workers;
        std::vector<std::vector<std::pair<float, bool>>> thread_distances(num_threads);
        
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            futures.push_back(std::async(std::launch::async, [&, thread_id]() {
                try {
                    size_t start = (data.size() * thread_id) / num_threads;
                    size_t end = (data.size() * (thread_id + 1)) / num_threads;
                    
                    for (size_t i = start; i < end; i += SAFE_BATCH_SIZE) {
                        size_t batch_end = std::min(i + SAFE_BATCH_SIZE, end);
                        
                        for (size_t j = i; j < batch_end; ++j) {
                            const auto& point = data[j];
                            
                            if (point.anchor.empty() || point.compare.empty()) {
                                continue;
                            }
                            
                            try {
                                std::vector<float> anchor_output = ind.evaluate(point.anchor);
                                std::vector<float> compare_output = ind.evaluate(point.compare);
                                
                                if (anchor_output.empty() || compare_output.empty()) {
                                    continue;
                                }

                                float distance = calculateDistance(anchor_output, compare_output);
                                
                                if (!std::isfinite(distance)) {
                                    continue;
                                }

                                thread_distances[thread_id].emplace_back(
                                    distance, point.label > 0.5f);
                                
                                float current_threshold = threshold_stats.getAdaptiveThreshold();
                                bool prediction = (distance < current_threshold);
                                bool is_same_class = (point.label > 0.5f);
                                
                                if (is_same_class) {
                                    ++total_positives;
                                    if (prediction) ++true_positives;
                                } else {
                                    ++total_negatives;
                                    if (!prediction) ++true_negatives;
                                }
                                
                                ++processed;
                                
                            } catch (const std::exception& e) {
                                continue;
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Thread " << thread_id << " error: " 
                            << e.what() << std::endl;
                }
            }));
        }
        
        for (auto& future : futures) {
            if (future.valid()) {
                try {
                    future.get();
                } catch (...) {
                    continue;
                }
            }
        }
        
        // Update threshold statistics with collected distances
        for (const auto& thread_dist : thread_distances) {
            for (const auto& [distance, is_same_class] : thread_dist) {
                threshold_stats.updateStats(distance, is_same_class);
            }
        }
        
        if (processed == 0) {
            return 0.0f;
        }

        float sensitivity = total_positives > 0 ? 
            static_cast<float>(true_positives) / total_positives : 0.0f;
        float specificity = total_negatives > 0 ? 
            static_cast<float>(true_negatives) / total_negatives : 0.0f;
        
        float balanced_accuracy = (sensitivity + specificity) / 2.0f;
        
        // Log stats every N generations or when in validation
        if (!ops->isTraining() || (processed % 1000) == 0) {
            std::cout << "Current threshold: " << threshold_stats.getAdaptiveThreshold()
                    << "\nPositive mean distance: " << threshold_stats.positive_mean
                    << "\nNegative mean distance: " << threshold_stats.negative_mean
                    << "\nSensitivity: " << (sensitivity * 100.0f) << "%"
                    << "\nSpecificity: " << (specificity * 100.0f) << "%"
                    << std::endl;
        }
        
        return balanced_accuracy;
    }
    
    Individual& runTournament(const std::vector<size_t>& available_indices) {
        if (available_indices.empty()) {
            throw std::runtime_error("No individuals available for tournament");
        }

        // Create a temporary vector for tournament candidates
        std::vector<size_t> tournament_candidates;
        tournament_candidates.reserve(std::min((size_t)config.tournament_size, available_indices.size()));

        // Create a copy of available indices to avoid modifying the original
        std::vector<size_t> available_copy = available_indices;
        
        // Use local RNG
        auto& local_gen = getGen();

        // Fill tournament pool without replacement
        while (tournament_candidates.size() < config.tournament_size && !available_copy.empty()) {
            std::uniform_int_distribution<size_t> idx_dist(0, available_copy.size() - 1);
            size_t random_idx = idx_dist(local_gen);
            
            // Add the selected index to tournament candidates
            tournament_candidates.push_back(available_copy[random_idx]);
            
            // Remove the selected index from available_copy
            std::swap(available_copy[random_idx], available_copy.back());
            available_copy.pop_back();
        }

        if (tournament_candidates.empty()) {
            throw std::runtime_error("Failed to select tournament candidates");
        }

        // Find the best individual in the tournament
        size_t best_idx = tournament_candidates[0];
        float best_fitness = std::numeric_limits<float>::max();

        // Safely find the best individual
        for (size_t idx : tournament_candidates) {
            if (idx < population.size()) {  // Bounds check
                float current_fitness = population[idx].getFitness();
                if (std::isfinite(current_fitness) && current_fitness < best_fitness) {
                    best_fitness = current_fitness;
                    best_idx = idx;
                }
            }
        }

        // Final safety check
        if (best_idx >= population.size()) {
            std::cerr << "Warning: Invalid best index " << best_idx 
                    << ", falling back to first individual" << std::endl;
            return population[0];  // Fallback to first individual
        }

        return population[best_idx];
    }
    
    float evaluateIndividual(const Individual& ind, const std::vector<DataPoint>& data) {
        try {
            // Check cache first
            float cached_fitness;
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (auto cached = fitness_cache.get(ind)) {
                    return *cached;
                }
            }

            float balanced_accuracy = calculateAccuracy(ind, data);
            
            if (!std::isfinite(balanced_accuracy)) {
                return std::numeric_limits<float>::max();
            }

            float complexity_penalty = config.parsimony_coeff * ind.totalSize();
            float fitness = (1.0f - balanced_accuracy) * config.fitness_alpha + complexity_penalty;

            if (std::isfinite(fitness)) {
                std::lock_guard<std::mutex> lock(cache_mutex);
                fitness_cache.put(ind, fitness);
            }

            return fitness;

        } catch (const std::exception& e) {
            std::cerr << "Unexpected error in evaluateIndividual: " << e.what() << std::endl;
            return std::numeric_limits<float>::max();
        }
    }

    void evaluatePopulation(const std::vector<DataPoint>& trainData) {
        if (population.empty() || trainData.empty()) {
            std::cerr << "Empty population or training data" << std::endl;
            return;
        }

        // Create vectors to store results safely
        std::vector<float> fitnesses(population.size(), std::numeric_limits<float>::max());
        std::vector<std::mutex> individual_mutexes(population.size());
        std::mutex global_mutex;

        // Process individuals in chunks
        const size_t chunk_size = std::max(size_t(1), population.size() / config.num_workers);
        std::vector<std::future<void>> futures;

        for (size_t start = 0; start < population.size(); start += chunk_size) {
            size_t end = std::min(start + chunk_size, population.size());
            
            futures.push_back(std::async(std::launch::async, [&, start, end]() {
                try {
                    for (size_t i = start; i < end; ++i) {
                        if (i >= population.size()) break;  // Safety check

                        // Evaluate individual
                        float fitness;
                        try {
                            {
                                std::lock_guard<std::mutex> lock(individual_mutexes[i]);
                                fitness = evaluateIndividual(population[i], trainData);
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "Error evaluating individual " << i << ": " << e.what() << std::endl;
                            continue;
                        }

                        // Store result if valid
                        if (std::isfinite(fitness)) {
                            std::lock_guard<std::mutex> lock(individual_mutexes[i]);
                            fitnesses[i] = fitness;
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Worker thread error: " << e.what() << std::endl;
                }
            }));
        }

        // Wait for all evaluations to complete
        for (auto& future : futures) {
            if (future.valid()) {
                try {
                    future.get();
                } catch (const std::exception& e) {
                    std::cerr << "Error waiting for future: " << e.what() << std::endl;
                }
            }
        }

        // Update population fitnesses and track best individual
        float best_fitness = std::numeric_limits<float>::max();
        size_t best_idx = 0;

        {
            std::lock_guard<std::mutex> lock(global_mutex);
            for (size_t i = 0; i < population.size(); ++i) {
                std::lock_guard<std::mutex> indiv_lock(individual_mutexes[i]);
                float current_fitness = fitnesses[i];
                
                try {
                    population[i].setFitness(current_fitness);
                    
                    if (current_fitness < best_fitness) {
                        best_fitness = current_fitness;
                        best_idx = i;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error updating fitness for individual " << i << ": " << e.what() << std::endl;
                }
            }
        }

        // Preserve best individual
        try {
            if (best_idx < population.size()) {
                std::lock_guard<std::mutex> lock(global_mutex);
                std::lock_guard<std::mutex> indiv_lock(individual_mutexes[best_idx]);
                Individual best_copy(population[best_idx]);  // Create copy of best individual
                population[0] = std::move(best_copy);       // Place at front of population
            }
        } catch (const std::exception& e) {
            std::cerr << "Error preserving best individual: " << e.what() << std::endl;
        }
    }

    std::vector<DataPoint> createStratifiedMinibatch(
        const std::vector<DataPoint>& trainData,
        size_t batch_size,
        std::mt19937& gen
    ) {
        // Separate positive and negative examples
        std::vector<const DataPoint*> positives;
        std::vector<const DataPoint*> negatives;
        
        for (const auto& point : trainData) {
            if (point.label > 0.5f) {
                positives.push_back(&point);
            } else {
                negatives.push_back(&point);
            }
        }
        
        // Calculate number of samples to take from each class
        // Try to maintain original class distribution but ensure at least 25% of minority class
        float pos_ratio = static_cast<float>(positives.size()) / trainData.size();
        pos_ratio = std::max(0.25f, std::min(0.75f, pos_ratio));  // Clamp between 25% and 75%
        
        size_t num_pos = static_cast<size_t>(batch_size * pos_ratio);
        size_t num_neg = batch_size - num_pos;
        
        // Adjust if we need more samples than available
        if (num_pos > positives.size()) {
            size_t diff = num_pos - positives.size();
            num_pos = positives.size();
            num_neg = std::min(negatives.size(), num_neg + diff);
        }
        if (num_neg > negatives.size()) {
            size_t diff = num_neg - negatives.size();
            num_neg = negatives.size();
            num_pos = std::min(positives.size(), num_pos + diff);
        }
        
        // Create minibatch with stratified sampling
        std::vector<DataPoint> batch;
        batch.reserve(num_pos + num_neg);
        
        // Sample positive examples
        if (!positives.empty()) {
            std::shuffle(positives.begin(), positives.end(), gen);
            for (size_t i = 0; i < num_pos; ++i) {
                batch.push_back(*positives[i % positives.size()]);
            }
        }
        
        // Sample negative examples
        if (!negatives.empty()) {
            std::shuffle(negatives.begin(), negatives.end(), gen);
            for (size_t i = 0; i < num_neg; ++i) {
                batch.push_back(*negatives[i % negatives.size()]);
            }
        }
        
        // Final shuffle of the batch
        std::shuffle(batch.begin(), batch.end(), gen);
        
        // Log class distribution in batch
        size_t batch_pos = std::count_if(batch.begin(), batch.end(), 
            [](const DataPoint& p) { return p.label > 0.5f; });
        float batch_pos_ratio = static_cast<float>(batch_pos) / batch.size();
        
        std::cout << "Minibatch class distribution:"
                << "\n  Total size: " << batch.size()
                << "\n  Positives: " << batch_pos 
                << " (" << (100.0f * batch_pos_ratio) << "%)"
                << "\n  Negatives: " << (batch.size() - batch_pos)
                << " (" << (100.0f * (1.0f - batch_pos_ratio)) << "%)" << std::endl;
        
        return batch;
    }
    
public:
    ContrastiveGP(const GPConfig& cfg) 
        : config(cfg)
        , ops(std::make_shared<GPOperations>(
            cfg.dropout_prob, cfg.bn_momentum, cfg.bn_epsilon)),
        threshold_stats(cfg.initial_distance_threshold)
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

    // In ContrastiveGP class, modify train() method:
    void train(const std::vector<DataPoint>& trainData, const std::vector<DataPoint>& valData) {
        if (trainData.empty() || valData.empty()) {
            return;
        }

        std::cout << "I get here I" << std::endl;

        // Use smaller batch size for faster iterations and more exploration
        // const size_t EVAL_SUBSET_SIZE = std::min(trainData.size(), trainData.size());
        const size_t EVAL_SUBSET_SIZE = std::min((size_t)1000, trainData.size());
        auto subset_gen = std::mt19937{std::random_device{}()};
        std::vector<size_t> tournament_indices(population.size());

        for (int generation = 0; generation < config.generations; ++generation) {
            std::vector<Individual> new_population;
            new_population.reserve(config.population_size);

            std::cout << "I get here II" << std::endl;

            try {
                // Create stratified training subset
                std::vector<DataPoint> train_subset = createStratifiedMinibatch(
                    trainData, EVAL_SUBSET_SIZE, subset_gen);

                // Evaluate current population
                if (!population.empty()) {
                    evaluatePopulation(train_subset);
                }

                std::cout << "I get here III" << std::endl;

                // Initialize tournament indices
                std::iota(tournament_indices.begin(), tournament_indices.end(), 0);

                // Add elite individuals first
                {
                    std::vector<std::pair<float, size_t>> sorted_indices;
                    sorted_indices.reserve(population.size());
                    
                    for (size_t i = 0; i < population.size(); ++i) {
                        sorted_indices.emplace_back(population[i].getFitness(), i);
                    }
                    
                    std::sort(sorted_indices.begin(), sorted_indices.end());
                    
                    for (size_t i = 0; i < config.elite_size && i < sorted_indices.size(); ++i) {
                        try {
                            size_t idx = sorted_indices[i].second;
                            if (idx < population.size()) {
                                new_population.push_back(Individual(population[idx]));
                            }
                        } catch (...) {
                            std::cout << "Error in elistim" << std::endl;
                            continue;
                        }
                    }
                }

                std::cout << "I get here IV" << std::endl;

                // Fill rest with tournament selection and crossover
                while (new_population.size() < config.population_size) {
                    try {
                        if (population.empty()) {
                            new_population.push_back(createRandomIndividual());
                            continue;
                        }

                        int num_parents = 2;
                        auto parents = selectParents(num_parents);
                        for (const auto& parent_pair : parents) {
                            Individual& parent1 = *parent_pair.first;
                            Individual& parent2 = *parent_pair.second;
                            
                            Individual offspring = crossover(parent1, parent2);

                            bool valid = true;
                            for (int i = 0; i < config.num_trees; i++) {
                                if (!offspring.getTree(i)) {
                                    valid = false;
                                    break;
                                }
                            }
                            
                            if (!valid) {
                                std::cout << "Invalid offspring created, using random individual instead" << std::endl;
                                offspring = createRandomIndividual();
                            }
                            
                            // Apply mutation with probability
                            if (std::uniform_real_distribution<float>{0.0f, 1.0f}(getGen()) < config.mutation_prob) {
                                offspring.mutate(getGen());
                            }
                            
                            new_population.push_back(std::move(offspring));
                        }
                    } catch (...) {
                        std::cout << "Error in crossover and mutation" << std::endl;
                        try {
                            new_population.push_back(createRandomIndividual());
                        } catch (...) {
                            continue;
                        }
                    }
                }

                std::cout << "I get here V" << std::endl;

                // Calculate statistics before population swap
                float gen_best_fitness = std::numeric_limits<float>::max();
                float train_acc = 0.0f;
                float val_acc = 0.0f;
                size_t best_idx = 0;
                
                // Find best individual and calculate accuracies
                if (!new_population.empty()) {
                    try {
                        for (size_t i = 0; i < new_population.size(); ++i) {
                            float fitness = new_population[i].getFitness();
                            if (fitness < gen_best_fitness) {
                                gen_best_fitness = fitness;
                                best_idx = i;
                            }
                        }

                        if (best_idx < new_population.size()) {
                            // Calculate accuracies on full 
                            if (!new_population[best_idx].getTree(0)) {
                                std::cerr << "Best individual is invalid" << std::endl;
                                continue;
                            }

                            // Calculate accuracies on full datasets
                            if (trainData.empty()) {
                                std::cerr << "Training data is empty" << std::endl;
                                continue;
                            }
                            std::cout << "Evaluating training accuracy" << std::endl;
                            train_acc = calculateAccuracy(new_population[best_idx], trainData);
                            ops->setTraining(false);
                            std::cout << "Evaluating test accuracy" << std::endl;
                            val_acc = calculateAccuracy(new_population[best_idx], valData);
                            ops->setTraining(true);
                        }
                    } catch (...) {
                        // If statistics calculation fails, use default values
                    }
                }

                std::cout << "I get here VI" << std::endl;

                // Safely swap populations
                if (!new_population.empty()) {
                    population = std::move(new_population);
                }

                // Print statistics
                std::cout << "Generation " << generation 
                        << "\n  Best Fitness: " << gen_best_fitness
                        << "\n  Train Accuracy: " << train_acc * 100.0f << "%"
                        << "\n  Val Accuracy: " << val_acc * 100.0f << "%"
                        << std::endl;

                printTreeDepthStats(population);

                // Clear cache periodically
                if (generation % 10 == 0) {
                    fitness_cache.clear();
                }

            } catch (...) {
                continue;
            }

            std::cout << "I get here VII" << std::endl;
        }
    }   
    
    Individual crossover(const Individual& parent1, const Individual& parent2) {
        // Create trees vector for offspring
        std::vector<std::unique_ptr<GPNode>> offspring_trees;
        offspring_trees.reserve(config.num_trees);
        
        int crossovers_performed = 0;
        int mutations_performed = 0;
        
        static thread_local std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        
        // Process each tree position
        for (int i = 0; i < config.num_trees; ++i) {
            try {
                // Get parent trees with safety checks
                auto* parent1_tree = parent1.getTree(i).get();
                auto* parent2_tree = parent2.getTree(i).get();
                
                if (!parent1_tree || !parent2_tree) {
                    // If either parent tree is invalid, create a new random tree
                    offspring_trees.push_back(
                        TreeOperations::createRandomOperatorNode(
                            config.max_tree_depth - 1, 0, config, ops, getGen()
                        )
                    );
                    continue;
                }
                
                // Decide whether to perform crossover
                bool should_crossover = prob_dist(getGen()) < config.crossover_prob;
                
                if (should_crossover && 
                    parent1_tree->isValidDepth() && 
                    parent2_tree->isValidDepth()) {
                    
                    // Attempt crossover
                    auto new_tree = TreeOperations::multiPointCrossover(
                        parent1_tree, parent2_tree, config, ops, getGen()
                    );
                    
                    // Validate result
                    if (new_tree && new_tree->isValidDepth()) {
                        offspring_trees.push_back(std::move(new_tree));
                        crossovers_performed++;
                        continue;
                    }
                }
                
                // If crossover fails or isn't chosen, try mutation
                if (auto new_tree = parent1_tree->clone()) {
                    TreeOperations::mutateNode(new_tree.get(), config, ops, getGen());
                    if (new_tree->isValidDepth()) {
                        offspring_trees.push_back(std::move(new_tree));
                        mutations_performed++;
                        continue;
                    }
                }
                
                // If all else fails, create random tree
                offspring_trees.push_back(
                    TreeOperations::createRandomOperatorNode(
                        config.max_tree_depth - 1, 0, config, ops, getGen()
                    )
                );
                
            } catch (...) {
                std::cout << "Error creating offspring tree" << std::endl;
                // On any error, add a random tree
                offspring_trees.push_back(
                    TreeOperations::createRandomOperatorNode(
                        config.max_tree_depth - 1, 0, config, ops, getGen()
                    )
                );
            }
        }
        
        // Create new individual with safety check
        try {
            return Individual(std::move(offspring_trees), ops, config);
        } catch (...) {
            std::cout << "Error creating offspring individual" << std::endl;
            // If individual creation fails, create a completely new random individual
            std::vector<std::unique_ptr<GPNode>> random_trees;
            random_trees.reserve(config.num_trees);
            
            for (int i = 0; i < config.num_trees; ++i) {
                random_trees.push_back(
                    TreeOperations::createRandomOperatorNode(
                        config.max_tree_depth - 1, 0, config, ops, getGen()
                    )
                );
            }
            
            return Individual(std::move(random_trees), ops, config);
        }
    }

    void printTreeDepthStats(const std::vector<Individual>& pop) {
        if (pop.empty()) {
            std::cout << "  Tree Depth Stats: No valid population" << std::endl;
            return;
        }

        try {
            struct DepthStats {
                int min_depth = std::numeric_limits<int>::max();
                int max_depth = 0;
                double avg_depth = 0.0;
                double std_dev = 0.0;
                std::vector<int> all_depths;  // Store all depths for variance calculation
            };

            // Calculate stats per tree position
            std::vector<DepthStats> stats_per_position(config.num_trees);
            
            // Overall stats across all trees
            DepthStats overall_stats;
            
            // Collect depths
            for (const auto& ind : pop) {
                for (int i = 0; i < config.num_trees; ++i) {
                    const auto& tree = ind.getTree(i);
                    if (tree) {
                        int depth = tree->depth();
                        if (depth > 0 && depth < 1000) {  // Sanity check
                            // Update per-position stats
                            stats_per_position[i].min_depth = std::min(stats_per_position[i].min_depth, depth);
                            stats_per_position[i].max_depth = std::max(stats_per_position[i].max_depth, depth);
                            stats_per_position[i].all_depths.push_back(depth);
                            
                            // Update overall stats
                            overall_stats.min_depth = std::min(overall_stats.min_depth, depth);
                            overall_stats.max_depth = std::max(overall_stats.max_depth, depth);
                            overall_stats.all_depths.push_back(depth);
                        }
                    }
                }
            }

            // Calculate averages and variances
            auto calculate_stats = [](DepthStats& stats) {
                if (stats.all_depths.empty()) return;
                
                // Calculate mean
                double sum = std::accumulate(stats.all_depths.begin(), stats.all_depths.end(), 0.0);
                stats.avg_depth = sum / stats.all_depths.size();
                
                // Calculate variance
                double variance = 0.0;
                for (int depth : stats.all_depths) {
                    double diff = depth - stats.avg_depth;
                    variance += diff * diff;
                }
                variance /= stats.all_depths.size();
                
                // Store standard deviation with the stats
                stats.std_dev = std::sqrt(variance);
            };

            // Calculate stats for each position and overall
            for (auto& stats : stats_per_position) {
                calculate_stats(stats);
            }
            calculate_stats(overall_stats);

            // Print statistics
            std::cout << "\nTree Depth Statistics:"
                    << "\n  Overall:"
                    << "\n    Min: " << overall_stats.min_depth
                    << "\n    Max: " << overall_stats.max_depth
                    << "\n    Avg: " << std::fixed << std::setprecision(2) << overall_stats.avg_depth
                    << "\n    Std: " << std::fixed << std::setprecision(2) << overall_stats.std_dev;

            // Print per-position statistics
            std::cout << "\n  Per Tree Position:";
            for (size_t i = 0; i < stats_per_position.size(); ++i) {
                const auto& stats = stats_per_position[i];
                if (!stats.all_depths.empty()) {
                    std::cout << "\n    Tree " << i << ":"
                            << " Min=" << stats.min_depth
                            << " Max=" << stats.max_depth
                            << " Avg=" << std::fixed << std::setprecision(2) << stats.avg_depth
                            << " Std=" << std::fixed << std::setprecision(2) << stats.std_dev;
                }
            }
            std::cout << std::endl;

        } catch (const std::exception& e) {
            std::cout << "Error calculating tree depth stats: " << e.what() << std::endl;
        }
    }
};

thread_local std::mt19937 GPOperations::rng = getThreadLocalRNG();
thread_local std::mt19937 ContrastiveGP::rng = getThreadLocalRNG();
size_t OperatorNode::next_node_id = 0;

// Example usage:
int main() {
    // Initialize configuration
    GPConfig config;

    std::cout << "Using " << config.num_workers << " worker threads" << std::endl;
    
    // Create and initialize model
    // ContrastiveGP model(config);
    auto ops = std::make_shared<GPOperations>(
            config.dropout_prob, config.bn_momentum, config.bn_epsilon);

    BinaryClassifier model(config, ops);
    
    // Read and process data
    ExcelProcessor processor;
    try {

        std::cout << "Reading the Excel file" << std::endl;
        auto filePath = "/vol/ecrg-solar/woodj4/fishy-business/data/REIMS.xlsx";
        // auto filePath = "/home/woodj/Desktop/fishy-business/data/REIMS.xlsx";
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