
#include <iostream>
#include <math.h>
#include <sstream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <ctime>
#include <chrono>

void populate_data_matrix(Eigen::MatrixXd *, std::vector<std::vector<double> >);
void get_classification(Eigen::VectorXd *, Eigen::VectorXd *);
double sigmoid(double);
void output_statistics(Eigen::MatrixXd);
void populate_labels(std::vector<std::vector<double> >, Eigen::VectorXd *, Eigen::VectorXd *);
void build_confusion_matrix(Eigen::MatrixXd *, Eigen::VectorXd, Eigen::VectorXd);
void populate_train_test(std::vector<std::vector<double> >, std::vector<std::vector<double> > *, std::vector<std::vector<double> > *);
void gradient_descent(double, Eigen::MatrixXd *, Eigen::VectorXd *, Eigen::VectorXd *);
void getProbs(Eigen::VectorXd *);

int main()
{
    std::fstream fin;
    fin.open("titanic_project.csv", std::ios::in);

    if (!fin.is_open())
    {
        std::cout << "Error opening file\n";
        return -1;
    }
    std::vector<std::vector<double> > data_frame;
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
        std::vector<double> vals;
        int flag = true;
        while (std::getline(s, val, ','))
        {

            double v;
            if (flag)
            {
                flag = false;
                std::string el = val.substr(1, val.size() - 2);
                v = std::stod(el);
            }
            else
            {
                v = std::stod(val);
            }
            vals.push_back(v);
        }
        data_frame.push_back(vals);
    }
    std::vector<std::vector<double> > train;
    std::vector<std::vector<double> > test;
    populate_train_test(data_frame, &train, &test);

    Eigen::VectorXd weights(2);
    Eigen::VectorXd train_labels(900);
    Eigen::VectorXd test_labels(data_frame.size() - 900);     //splitting our data set into train and test sets
    populate_labels(data_frame, &test_labels, &train_labels); //filling up the survived vectors for both training and testing

    for (int i = 0; i < 2; i++)
    { //initializing weights to 1.0
        weights(i) = 1.0;
    }
    Eigen::MatrixXd data_matrix(train.size(), 2);
    Eigen::MatrixXd test_data_matrix(test.size(), 2);
    populate_data_matrix(&data_matrix, train); //populating the eigen matrix with the data from the vector
    populate_data_matrix(&test_data_matrix, test);
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    gradient_descent(.001, &data_matrix, &weights, &train_labels); //training
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);
    std::cout << "Time taken: " << time_span.count() << std::endl;
    std::cout << "Weights: " << std::endl;
    std::cout << weights << std::endl;
    Eigen::VectorXd probs = test_data_matrix * weights;
    getProbs(&probs);
    Eigen::VectorXd results(probs.size());
    get_classification(&probs, &results);
    Eigen::MatrixXd confusion_matrix(2, 2);
    build_confusion_matrix(&confusion_matrix, results, test_labels);
    std::cout << "Confusion matrix key: (0,0) = true survived, (1,1) = true died" << std::endl;
    std::cout << confusion_matrix << std::endl;
    output_statistics(confusion_matrix);
}

double sigmoid(double val)
{
    return 1.0 / (1.0 + exp(-val));
}
void output_statistics(Eigen::MatrixXd confusion_matrix)
{
    std::cout << "Accuracy: " << (confusion_matrix(0, 0) + confusion_matrix(1, 1)) / (confusion_matrix(0, 0) + confusion_matrix(1, 0) + confusion_matrix(0, 1) + confusion_matrix(1, 1)) << std::endl;
    std::cout << "Sensitivy: " << confusion_matrix(0, 0) / (confusion_matrix(0, 0) + confusion_matrix(1, 0)) << std::endl;
    std::cout << "Specificity: " << confusion_matrix(1, 1) / (confusion_matrix(1, 1) + confusion_matrix(0, 1)) << std::endl;
    ;
}
void populate_labels(std::vector<std::vector<double> > data, Eigen::VectorXd *test_label, Eigen::VectorXd *train_label)
{
    int test_iter = 0;
    for (int i = 0; i < data.size(); i++)
    {

        if (i < 900)
        {
            (*train_label)(i) = data[i][2];
        }
        else
        {

            (*test_label)(test_iter) = data[i][2];
            test_iter++;
        }
    }
}
void populate_train_test(std::vector<std::vector<double> > data, std::vector<std::vector<double> > *train, std::vector<std::vector<double> > *test)
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
void populate_data_matrix(Eigen::MatrixXd *data_matrix, std::vector<std::vector<double> > train)
{ // only need to populate two columns, the intercept and pclass

    for (int i = 0; i < train.size(); i++)
    {

        (*data_matrix)(i, 0) = 1;

        (*data_matrix)(i, 1) = train[i][1];
    }
}

void gradient_descent(double learning_rate, Eigen::MatrixXd *data_matrix, Eigen::VectorXd *weights, Eigen::VectorXd *labels)
{
    //std::cout << *weights << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        Eigen::VectorXd probs = (*data_matrix) * (*weights);

        getProbs(&probs);
        Eigen::VectorXd error = (*labels) - probs;

        (*weights) = (*weights) + learning_rate * (data_matrix->transpose() * error);
    }
}

void getProbs(Eigen::VectorXd *probs)
{
    for (int i = 0; i < probs->size(); i++)
    {
        (*probs)(i) = sigmoid((*probs)(i));
    }
}
void get_classification(Eigen::VectorXd *probs, Eigen::VectorXd *results)
{
    for (int i = 0; i < probs->size(); i++)
    {
        (*results)(i) = (*probs)(i) > .5 ? 1 : 0;
    }
}
void build_confusion_matrix(Eigen::MatrixXd *matrix, Eigen::VectorXd p, Eigen::VectorXd actual)
{
    int true_positive = 0;
    int true_negative = 0;
    int false_positive = 0;
    int false_negative = 0;
    Eigen::VectorXd res = actual - p;
    for (int i = 0; i < p.size(); i++)
    {
        if (res(i) < .001)
        { //was correct, using epsilon comparison because they are both doubles
            if (p(i) > .5)
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
            if (p(i) > .5)
            {
                false_positive++;
            }
            else
            {
                false_negative++;
            }
        }
    }

    (*matrix)(0, 0) = (double)true_positive;
    (*matrix)(0, 1) = (double)false_positive;
    (*matrix)(1, 0) = (double)false_negative;
    (*matrix)(1, 1) = (double)true_negative;
}
