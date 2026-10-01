#ifndef MODEL_BUILDER_H
#define MODEL_BUILDER_H

#include "DataSet/DataSet.h"

#include <string>
#include <vector>
#include <set>
#include "ErrorStrings.h"

#ifdef USE_WEBSOCKET_LOGGING
#include "WebSocketLogger/WebSocketLogger.h"
#endif

using namespace std;


/**
 * @class ModelBuilder
 * @brief Class for handling conditional inference tree data processing.
 */
class ModelBuilder {
private:
    string dataDirectory;  ///< Directory containing data files.
    int totalColumns;           ///< Total number of columns in data.
    int totalRows;              ///< Total number of rows in data.
    int numFiles;               ///< Number of files to read.
    int *targetColumnFlags;      //bitwise flag to indicate which columns will be used as target columns
    int *featureColumnFlags;     //bitwise flag to indicate which columns will be used as feature columns
    double alpha;                  ///< Alpha value for significance testing.

    #ifdef USE_WEBSOCKET_LOGGING
    unique_ptr<WebSocketLogger> webSocketLogger;
    int generalLogChannelKey; ///< Key for the general log channel
    #endif

public:
    /**
     * @brief Constructor initializes variables.
     */
    ModelBuilder();

    /**
     * @brief Destructor to free allocated memory.
     */
    ~ModelBuilder();

    #ifdef USE_WEBSOCKET_LOGGING
    /**
     * @brief Initializes the WebSocket logger with the given server URL.
     * @param serverUrl The URL of the WebSocket server.
     */
    void initializeWebSocketLogger(const string& serverUrl);
    #endif

    /**
     * @brief Sets the data directory and reads metadata from meta.txt.
     * @param dir The directory containing the meta.txt file.
     */
    void setDataDirectory(const string& dir);

    /**
     * @brief Sets the alpha value for significance testing.
     * @param alpha The alpha value to set.
     */
    void setAlpha(double alpha);

    /**
     * @brief Enables a column for use as a target column.
     * @param columnNum The column number to enable.
     */
    void enableColumnForTarget(int columnNum);

    /**
     * @brief Enables a column for use as a feature column.
     * @param columnNum The column number to enable.
     */
    void enableColumnForFeature(int columnNum);

    /**
     * @brief Starts the process of building the model.
     */
    void buildModel();

    /**
     * @brief Returns the total number of columns in the data.
     * @return The total number of columns.
     */
    int getTotalDataSetColumns();

    /**
     * @brief Checks if a feature column is enabled.
     * @param columnNum The column number to check.
     * @return True if the column is enabled, false otherwise.
     */
    bool isFeatureColumnEnabled(int columnNum);

    /**
     * @brief Checks if a column is enabled to be used as a target column.
     * @param columnNum The column number to check.
     * @return True if the column is enabled, false otherwise.
     */
    bool isTargetColumnEnabled(int columnNum);

    /**
    * @brief The method below is only for testing purposes
    */
    void printData();

private:
    /**
     * @brief Creates a tree for a given target column.
     * @param targetColumn The target column to create the tree for.
     */
    void createTree(int targetColumn);

};

#endif // MODEL_BUILDER_H
