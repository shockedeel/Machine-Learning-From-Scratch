#include <iostream>
#include <fstream>
#include <math.h>
#include <sstream>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_map>
#include <Eigen/Dense>

void populate_train_test(std::vector<std::vector<int> >, std::vector<std::vector<int> > *, std::vector<std::vector<int> > *);

class NaiveBayes
{
    std::vector<std::vector<int> > data;
    std::vector<int> outcomes;
    std::vector<int> sizeOutcomes;
    Eigen::MatrixXd aPriori;
    Eigen::MatrixXd pclassLikelihood;
    Eigen::MatrixXd sexLikelihood;
    Eigen::MatrixXd ageLikelihood;
    std::chrono::duration<double> training_time;

public:
    NaiveBayes(std::vector<std::vector<int> > d)
    {
        data = d;
        aPriori = Eigen::MatrixXd(1, 2);
        pclassLikelihood = Eigen::MatrixXd(2, 3);
        sexLikelihood = Eigen::MatrixXd(2, 2);
        ageLikelihood = Eigen::MatrixXd(2, 2);
        trainModel();
    }
    void outputInfo()
    {
        std::cout << "training time: " << training_time.count() << std::endl;
        std::cout << "a priori: " << std::endl
                  << aPriori << std::endl;
        std::cout << "conditional probabilities:" << std::endl
                  << "pclass" << std::endl
                  << pclassLikelihood << std::endl;
        std::cout << "sex" << std::endl
                  << sexLikelihood << std::endl;
        std::cout << "age" << std::endl
                  << ageLikelihood << std::endl;
    }
    Eigen::Vector2d getRawProbability(int pclass, int sex, int age)
    {

        double num_s = pclassLikelihood(1, pclass - 1) * sexLikelihood(1, sex) * aPriori(0, 1) * getContinuousLikelihood(age, 1, ageLikelihood);
        double num_p = pclassLikelihood(0, pclass - 1) * sexLikelihood(0, sex) * aPriori(0, 0) * getContinuousLikelihood(age, 0, ageLikelihood);
        double denominator = num_s + num_p;

        Eigen::Vector2d raw_probs(2);
        raw_probs(0) = num_p / denominator;
        raw_probs(1) = num_s / denominator;

        return raw_probs;
    }
    int getClassification(Eigen::Vector2d vector)
    {
        if (vector(0) > .5)
        {
            return 0;
        }
        return 1;
    }
    std::vector<int> getPredictions(std::vector<std::vector<int> > data)
    {
        std::vector<int> preds;
        for (int i = 0; i < data.size(); i++)
        {

            preds.push_back(getClassification(getRawProbability(data[i][1], data[i][3], data[i][4])));
        }
        Eigen::Matrix2d confusion = generateConfusionMatrix(preds, data);
        outputStatistics(confusion);
        return preds;
    }

private:
    void trainModel()
    {
        outcomes.push_back(0);
        outcomes.push_back(1);
        std::vector<int> pclass_vals;
        pclass_vals.push_back(1);
        pclass_vals.push_back(2);
        pclass_vals.push_back(3);
        std::vector<int> sex_vals;
        sex_vals.push_back(0);
        sex_vals.push_back(1);
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        calculateAPriori();
        calculateDiscreteLikelihoodMatrix(pclass_vals, &pclassLikelihood, 1);
        calculateDiscreteLikelihoodMatrix(sex_vals, &sexLikelihood, 3);
        calculateContinuousLikelihoodMatrix(&ageLikelihood, 4);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        training_time = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);
    }
    void calculateAPriori()
    {
        int died = 0;
        int survived = 0;
        for (int i = 0; i < data.size(); i++)
        {
            if (data[i][2] == 1)
            {
                survived++;
            }
            else
            {
                died++;
            }
        }
        sizeOutcomes.push_back(died);
        sizeOutcomes.push_back(survived);
        aPriori(0, 0) = (double)died / data.size();
        aPriori(0, 1) = (double)survived / data.size();
    }
    void outputStatistics(Eigen::Matrix2d confusion)
    {
        double sensitivity = confusion(1, 1) / (confusion(1, 1) + confusion(0, 1));
        double specificity = confusion(0, 0) / (confusion(0, 0) + confusion(1, 0));
        double accuracy = (confusion(0, 0) + confusion(1, 1)) / (confusion(0, 0) + confusion(1, 1) + confusion(0, 1) + confusion(1, 0));
        std::cout << "sensitivity: " << sensitivity << std::endl;
        std::cout << "specificity: " << specificity << std::endl;
        std::cout << "accuracy: " << accuracy << std::endl;
        std::cout << "confusion matrix: (0,0)= true perished(true negative) (1,1) = true survived(true positive)" << std::endl
                  << confusion << std::endl;
    }
    Eigen::Matrix2d generateConfusionMatrix(std::vector<int> predictions, std::vector<std::vector<int> > data)
    {
        Eigen::Matrix2d confusion;
        int true_positive = 0, true_negative = 0, false_positive = 0, false_negative = 0;
        if (predictions.size() != data.size())
            return confusion;
        for (int i = 0; i < data.size(); i++)
        {
            if (predictions[i] == data[i][2])
            { //was correct
                if (predictions[i] == 1)
                {
                    true_positive++;
                }
                else
                {
                    true_negative++;
                }
            }
            else
            { //was incorrect
                if (predictions[i] == 1)
                {
                    false_positive++;
                }
                else
                {
                    false_negative++;
                }
            }
        }

        confusion(0, 0) = true_negative;
        confusion(0, 1) = false_negative;
        confusion(1, 0) = false_positive;
        confusion(1, 1) = true_positive;
        return confusion;
    }
    void calculateContinuousLikelihoodMatrix(Eigen::MatrixXd *matrix, int target_col)
    {

        for (int i = 0; i < outcomes.size(); i++)
        {
            double mean = calculateMean(target_col, data, outcomes[i], 2);
            double variance = calculateVariance(target_col, data, outcomes[i], 2);
            (*matrix)(i, 0) = mean;
            (*matrix)(i, 1) = variance;
        }
    }
    void calculateDiscreteLikelihoodMatrix(std::vector<int> possible_values, Eigen::MatrixXd *matrix, int data_col)
    {

        for (int i = 0; i < outcomes.size(); i++)
        {
            std::unordered_map<int, int> class_counts; //hashmap to count the number of appearances per class
            for (int j = 0; j < data.size(); j++)
            {
                if (data[j][2] != outcomes[i])
                    continue;
                if (class_counts.find(data[j][data_col]) == class_counts.end())
                {

                    class_counts[data[j][data_col]] = 1;
                }
                else
                {
                    class_counts[data[j][data_col]]++;
                }
            }

            for (int j = 0; j < possible_values.size(); j++)
            {

                (*matrix)(i, j) = (double)class_counts[possible_values[j]] / (double)sizeOutcomes[i]; //using the class counts to find out conditional probs
            }
        }
    }

    double getContinuousLikelihood(int x, int type, Eigen::MatrixXd likelihood_mat)
    {

        return (1.0 / sqrt(2 * M_PI * likelihood_mat(type, 1))) * exp(-((pow(x - likelihood_mat(type, 0), 2))) / (2 * likelihood_mat(type, 1)));
    }
    double calculateMean(int target_col, std::vector<std::vector<int> > data, int c, int class_col)
    {
        int n = 0;
        int sum = 0;
        for (int i = 0; i < data.size(); i++)
        {
            if (data[i][class_col] != c)
                continue;
            sum += data[i][target_col];
            n++;
        }
        return (double)sum / (double)n;
    }
    double calculateVariance(int target_col, std::vector<std::vector<int> > data, int c, int class_col)
    {
        double mean = calculateMean(target_col, data, c, class_col);
        double sum = 0.0;
        int n = 0;
        for (int i = 0; i < data.size(); i++)
        {
            if (data[i][class_col] != c)
                continue;
            n++;
            sum += pow((double)data[i][target_col] - mean, 2);
        }
        return pow(sum / (double)n, .5);
    }
};

int main()
{
    std::fstream fin;
    fin.open("titanic_project.csv", std::ios::in);

    if (!fin.is_open())
    {
        std::cout << "Error opening file\n";
        return -1;
    }
    std::vector<std::vector<int> > data_frame;
    std::string line;
    int firstRow = true;
    while (fin >> line)
    {
        if (firstRow)
        {
            firstRow = false;
            continue;
        }

        std::stringstream s(line);
        std::string val;
        std::vector<int> vals;
        int flag = true;
        while (std::getline(s, val, ','))
        {

            int v;
            if (flag)
            {
                flag = false;
                std::string el = val.substr(1, val.size() - 2);
                v = std::stoi(el);
            }
            else
            {
                v = std::stod(val);
            }
            vals.push_back(v);
        }
        data_frame.push_back(vals);
    }
    std::vector<std::vector<int> > train;
    std::vector<std::vector<int> > test;
    populate_train_test(data_frame, &train, &test);
    NaiveBayes model = NaiveBayes(train);
    model.outputInfo();
    model.getPredictions(test);
}

void populate_train_test(std::vector<std::vector<int> > data, std::vector<std::vector<int> > *train, std::vector<std::vector<int> > *test)
{
    for (int i = 0; i < data.size(); i++)
    {
        if (i < 900)
        {
            train->push_back(data[i]);
        }
        else
        {
            test->push_back(data[i]);
        }
    }
}
