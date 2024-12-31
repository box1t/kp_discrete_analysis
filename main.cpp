#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <limits>
#include <cmath>

class TextPreprocessor {
public:
    std::vector<std::string> preprocess_text(const std::string& input_text) {
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

        return text_pieces;
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
    void train(const std::vector<std::string>& documents, const std::vector<std::string>& class_labels) {
        for (size_t i = 0; i < documents.size(); ++i) {
            const auto& doc = documents[i];
            const auto& class_label = class_labels[i];

            ++documents_in_classes_counter[class_label];
            TextPreprocessor text_preprocessor;
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

    std::string classify(const std::string& document) const {
        TextPreprocessor text_preprocessor;
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

    NaiveBayesModel classifier;
    classifier.train(train_documents, train_class_labels);

    for (int i = 0; i < test_counter; ++i) {
        std::string testDocument;
        std::getline(std::cin, testDocument);

        std::cout << classifier.classify(testDocument) << std::endl;
        std::cout.flush(); 
    }

    return 0;
}