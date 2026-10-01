#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <string>
#include <chrono>

using namespace std;

class ProgressBar {
public:
    enum class Color {
        DEFAULT,
        RED,
        GREEN,
        YELLOW,
        BLUE,
        MAGENTA,
        CYAN,
        WHITE
    };

    ProgressBar();
    void startTracking(string label, long long numSteps, Color barColor = Color::DEFAULT);
    void update(long long currentStep);
    void complete();

private:
    long long  totalSteps;
    Color barColor;
    string label;
    chrono::time_point<chrono::high_resolution_clock> startTime;
    chrono::time_point<chrono::high_resolution_clock> endTime;

    void display(long long currentStep, float progress) const;
    string getColorCode(Color color) const;
    void resetColor() const;
};

#endif // PROGRESS_BAR_H
