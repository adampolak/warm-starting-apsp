#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
using namespace std;

typedef vector<vector<int>> Matrix;
typedef vector<vector<vector<int>>> Prediction;


Matrix read_matrix(const std::string& path) {
    ifstream fin(path);
    Matrix matrix;    
    std::string line;
    while (std::getline(fin, line)) {
        istringstream iss(line);
        vector<int> row;
        float value;
        while (iss >> value) {
            row.push_back(int(1000*value));
        }
        matrix.push_back(std::move(row));
    }
    return matrix;
}


Matrix floyd_warshall(Matrix G) {
    int n = G.size();
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                G[i][j] = min(G[i][j], G[i][k] + G[k][j]);
    return G;
}


Prediction prepare_prediction(const Matrix& D, int q) {
    int n = D.size();
    Prediction prediction(n, vector<vector<int>>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            vector<pair<int, int>> midpoints(n);
            for (int k = 0; k < n; ++k) {
                midpoints[k].first = D[i][k] + D[k][j];
                midpoints[k].second = k;
            }
            sort(midpoints.begin(), midpoints.end());
            for (int t = 0; t < min(q, n); ++t)
                prediction[i][j].push_back(midpoints[t].second);
            sort(prediction[i][j].begin(), prediction[i][j].end());
        }
    }
    return prediction;
}


Matrix simul_dijkstra(Matrix G, const Prediction& prediction) {
    int n = G.size();
    Prediction pred_i(n, vector<vector<int>>(n));
    Prediction pred_j(n, vector<vector<int>>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k : prediction[i][j]) {
                pred_i[k][i].push_back(j);
                pred_j[k][j].push_back(i);
            }
        }
    }

    using State = tuple<int, int, int>;
    priority_queue<State, vector<State>, greater<State>> pq;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                pq.emplace(G[i][j], i, j);
            }
        }
    }

    while (!pq.empty()) {
        const auto [dist, i, j] = pq.top();
        pq.pop();
        if (dist > G[i][j]) continue;
        for (int k : pred_i[j][i]) {
            int new_val = G[i][j] + G[j][k];
            if (new_val < G[i][k]) {
                G[i][k] = new_val;
                pq.emplace(new_val, i, k);
            }
        }
        for (int k : pred_j[i][j]) {
            int new_val = G[k][i] + G[i][j];
            if (new_val < G[k][j]) {
                G[k][j] = new_val;
                pq.emplace(new_val, k, j);
            }
        }
    }
    return G;
}


int pred_error(const Matrix& D, const Matrix& true_D, const Prediction& pred, int p) {
    int n = D.size();
    int error = 0;

    auto detour = [&](int u, int v, int w) {
        return D[u][w] + D[w][v] - D[u][v];
    };

    for (int u = 0; u < n; ++u) {
        for (int v = 0; v < n; ++v) {
            if (D[u][v] != true_D[u][v]) {
                ++error;
                continue;
            }
            vector<int> in_cert;
            int min_out_cert = (int)1e9;
            for (int w = 0; w < n; ++w) {
                if (binary_search(pred[u][v].begin(), pred[u][v].end(), w))
                    in_cert.push_back(detour(u, v, w));
                else
                    min_out_cert = min(min_out_cert, detour(u, v, w));
            }

            sort(in_cert.begin(), in_cert.end());
            if (in_cert[p - 1] > min_out_cert)
                ++error;
        }
    }
    return error;
}


void experiment(string path_prefix, int num_graphs) {
    cout << path_prefix << "*" << endl;
    vector<Matrix> graphs;
    for (int i = 1; i <= num_graphs; ++i)
        graphs.push_back(read_matrix(path_prefix + to_string(i)));
    int n = graphs[0].size();
    int q = round(2.0 * sqrt(n));
    int p = round(0.5 * sqrt(n));
    cout << "n=" << n << " q=" << q << " p=" << p << endl << endl;
    const int NUM_RUNS = 10;
    vector<int> errors;
    for (int i = 0; i < NUM_RUNS; ++i) {
        int a = rand() % (graphs.size() - 1);
        int b = a + 1;
        Matrix Da = floyd_warshall(graphs[a]);
        Matrix true_Db = floyd_warshall(graphs[b]);
        Prediction P = prepare_prediction(Da, q);
        Matrix estimate_Db = simul_dijkstra(graphs[b], P);
        errors.push_back(pred_error(estimate_Db, true_Db, P, p));
        cout << "Run " << i + 1 << "/" << NUM_RUNS << ": error=" << errors.back() << endl;
    }
    cout << endl;
    double mean_error = accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    double sq_sum = 0.0;
    for (int error : errors) {
        double d = error - mean_error;
        sq_sum += d * d;
    }
    double stddev = sqrt(sq_sum / errors.size());
    cout << "Mean error = " << mean_error << " stddev = " << stddev << endl << endl;
}


int main() {
    experiment("3rdparty/NetLatency-Data/Seattle/SeattleData_", 688);
    experiment("3rdparty/NetLatency-Data/PlanetLab/PlanetLabData_", 18);
}
