#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <limits>
#include <cmath>
#include <chrono>

class TextPreprocessor {
private:
    int callCount = 0;
    double totalPreprocessTime = 0.0;

public:
    std::vector<std::string> preprocess_text(const std::string& input_text) {
        auto preprocessStart = std::chrono::high_resolution_clock::now();

        ++callCount;
        std::vector<std::string> text_pieces;
        std::string text_piece;

        for (char ch : input_text) {
            if (std::isspace(ch)) {
                if (!text_piece.empty()) {
                    text_pieces.push_back(std::move(text_piece));
                    text_piece.clear();
                }
            } else {
                text_piece += std::tolower(ch);
            }
        }

        if (!text_piece.empty()) {
            text_pieces.push_back(std::move(text_piece));
        }

        auto preprocessEnd = std::chrono::high_resolution_clock::now();
        totalPreprocessTime += std::chrono::duration<double>(preprocessEnd - preprocessStart).count();

        return text_pieces;
    }

    int getCallCount() const {
        return callCount;
    }

    double getTotalPreprocessTime() const {
        return totalPreprocessTime;
    }
};

class NaiveBayesModel {
private:
    std::unordered_map<std::string, std::unordered_map<std::string, int>> word_frequencies_in_classes;
    std::unordered_map<std::string, int> documents_in_classes_counter;
    std::unordered_map<std::string, int> total_words_in_classes_counter;
    std::unordered_map<std::string, double> log_prior_prob_for_classes;
    double smoothing_factor = 0.01;

public:
    void train(const std::vector<std::string>& documents, const std::vector<std::string>& class_labels, TextPreprocessor& text_preprocessor) {
        for (size_t i = 0; i < documents.size(); ++i) {
            const auto& doc = documents[i];
            const auto& class_label = class_labels[i];

            ++documents_in_classes_counter[class_label];
            for (const auto& text_piece : text_preprocessor.preprocess_text(doc)) {
                ++word_frequencies_in_classes[text_piece][class_label];
                ++total_words_in_classes_counter[class_label];
            }
        }

        int total_docs = documents.size();
        for (const auto& [class_label, count] : documents_in_classes_counter) {
            log_prior_prob_for_classes[class_label] = std::log(static_cast<double>(count)) - std::log(total_docs);
        }
    }

    std::string classify(const std::string& document, TextPreprocessor& text_preprocessor) const {
        std::vector<std::string> text_pieces = text_preprocessor.preprocess_text(document);

        std::unordered_map<std::string, int> text_pieces_counts_in_doc;
        for (const auto& text_piece : text_pieces) {
            ++text_pieces_counts_in_doc[text_piece];
        }

        double best_score = -std::numeric_limits<double>::infinity();
        std::string best_class;

        for (const auto& class_entry : documents_in_classes_counter) {
            const std::string& class_label = class_entry.first;
            double score = log_prior_prob_for_classes.at(class_label);

            for (const auto& [text_piece, freq_in_doc] : text_pieces_counts_in_doc) {
                int token_count_for_class = 0;
                if (word_frequencies_in_classes.count(text_piece) 
                && word_frequencies_in_classes.at(text_piece).count(class_label)) 
                {
                    token_count_for_class = word_frequencies_in_classes.at(text_piece).at(class_label);
                }

                double numerator = token_count_for_class + smoothing_factor;
                double denominator = total_words_in_classes_counter.at(class_label) + smoothing_factor;
                score += freq_in_doc * (std::log(numerator) - std::log(denominator));
            }

            if (score > best_score) {
                best_score = score;
                best_class = class_label;
            }
        }

        return best_class;
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int train_counter = 0, test_counter = 0;

    std::cin >> train_counter >> test_counter;
    std::cin.get(); 

    std::vector<std::string> train_documents(train_counter);
    std::vector<std::string> train_class_labels(train_counter);

    for (int i = 0; i < train_counter; ++i) {
        std::getline(std::cin, train_class_labels[i]);
        std::getline(std::cin, train_documents[i]);
    }

    TextPreprocessor text_preprocessor;
    NaiveBayesModel classifier;

    auto trainStart = std::chrono::high_resolution_clock::now();
    classifier.train(train_documents, train_class_labels, text_preprocessor);
    auto trainEnd = std::chrono::high_resolution_clock::now();
    double trainTime = std::chrono::duration<double>(trainEnd - trainStart).count();

    auto predictStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_counter; ++i) {
        std::string testDocument;
        std::getline(std::cin, testDocument);
        std::cout << classifier.classify(testDocument, text_preprocessor) << std::endl;
    }
    auto predictEnd = std::chrono::high_resolution_clock::now();
    double predictTime = std::chrono::duration<double>(predictEnd - predictStart).count();

    std::cerr << "[BENCHMARK] Train time: " << trainTime << " s\n";
    std::cerr << "[BENCHMARK] Classify time (all test docs): " << predictTime << " s\n";
    std::cerr << "[BENCHMARK] Preprocess call count: " << text_preprocessor.getCallCount() << "\n";
    std::cerr << "[BENCHMARK] Total preprocessing time: " << text_preprocessor.getTotalPreprocessTime() << " s\n";

    return 0;
}
