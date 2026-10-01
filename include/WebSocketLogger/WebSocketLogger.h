#ifndef WEBSOCKETLOGGER_H
#define WEBSOCKETLOGGER_H

#include <ixwebsocket/IXWebSocket.h>
#include <string>
#include <unordered_map>
#include <mutex>

class WebSocketLogger {
public:
    WebSocketLogger(const std::string& serverUrl);
    ~WebSocketLogger();

    // Setup
    int createProgressBar(const std::string& title, int totalIterations);
    int createLogChannel(const std::string& title);

    // Usage
    void updateStatus(int progressBarKey, int iterationValue);
    void completeProgressBar(int progressBarKey);
    void log(int logChannelKey, const std::string& message);
    void clearAll();

private:
    ix::WebSocket webSocket;
    std::mutex sendMutex;

    std::unordered_map<std::string, int> progressTitleToKey;
    std::unordered_map<std::string, int> logTitleToKey;

    int generateUniqueKey(const std::unordered_map<std::string, int>& existingKeys);
    void sendJson(const std::string& json);
};

#endif // WEBSOCKETLOGGER_H
