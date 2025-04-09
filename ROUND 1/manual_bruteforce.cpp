#include <iostream>
#include <iomanip>
#include <vector>
#include <cstddef>

using namespace std;

const int NUM_COMMODITIES = 4;
const int MAX_PATH_LEN = 5;

const vector<vector<double>> exchangeRates = {
    {1.0, 1.45, 0.52, 0.72},
    {0.7, 1.0, 0.31, 0.48},
    {1.95, 3.1, 1.0, 1.49},
    {1.34, 1.98, 0.64, 1.0}
};

vector<int> best_path;
double best_factor = 0.0;

void dfs(int current_index, double current_factor, vector<int>& path, int depth) {
    if (depth > MAX_PATH_LEN) return;

    if (current_index == 3 && depth > 0) {
        if (current_factor > best_factor) {
            best_factor = current_factor;
            best_path = path;
        }
    }

    for (int next = 0; next < NUM_COMMODITIES; ++next) {
        path.push_back(next);
        dfs(next, current_factor * exchangeRates[current_index][next], path, depth + 1);
        path.pop_back();
    }
}

int main() {
    vector<int> path;
    path.push_back(3); // Start from currency (index 3)
    dfs(3, 1.0, path, 0);

    cout << fixed << setprecision(5);
    cout << "Best Exchange Factor: " << best_factor << endl;
    cout << "Best Path: ";
    for (int i : best_path) cout << i << " ";
    cout << endl;

    return 0;
}
