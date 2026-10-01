#ifndef PARSER_H
#define PARSER_H

#include <string>
#include <unordered_map>
#include <fstream>
#include <vector>
// #include <FlexLexer.h>

// Include the Flex-generated header


using namespace std;

#define columnMapDir "column_maps"
#define mapCountFile "int_map_count.json"
#define strIntMapFile "str_int_map.json"



#define NUM_THREADS 8 // Number of threads for multi-threading
#define COLUMN_GROUP_SIZE 1000 // Number of columns to process in one group




typedef struct ColumnGroupContext {
    int numColumns;
    int numRows;
    int bitWidth;
    int columnGroupIndex;
    int **mappedValues;
} ColumnGroupContext;

typedef struct ReadBinFileContext {
    string fileName;    // Input file stream
    int numRows;      // Number of rows in the file
    int numColumns;   // Number of columns in the file
    int bitWidth;      // Number of bits needed for mapping
    int columnGroupIndex;
} ReadBinFileContext;

typedef struct FileContext {
    string fileName;  // File name
    string fileDataDirectory; // Directory for file data
    string mapCountDirectory; // Directory for map counts
    int fileIndex;    // Index of the file in the list of input files
    long long int fileSize;
    long long int* fileOffsets; // Array of byte offsets for each line
    int numRows;      // Total number of rows (observations) in the file
    int numColumns;   // Total number of columns in the file
    int largestMappedValue;   // Largest mapped value for the file
    int numBits;              // Number of bits needed for mapping
    int numColumnGroups;      // Number of column groups
    vector<vector<int>> *mappedValueCounts; // Vector of counts for each mapped value
    vector<unordered_map<string, int>> *columnMaps; // Vector of maps for each column
    int **mappedValues; // 2D array for mapped integers [numRows][numColumns]          
} FileContext;


/**
 * @brief The Parser class is responsible for parsing CSV files and mapping string values to integers.
 */
class Parser {
public:
    // Constructors and Destructors
    Parser();
    ~Parser();

    // Public Methods

    /**
    * @brief Adds an input file to the list of files and processes it.
    *
    * This function checks if the specified file exists and is readable. It then adds the file to the list of input files,
    * sets the current file context, ensures the file ends with a newline, creates necessary output directories,
    * and processes the file to determine the number of rows and columns. Finally, it parses the rows in parallel.
    *
    * @param fileName The name of the file to be added and processed.
    * @throws runtime_error If the file does not exist or cannot be read, or if the output directory cannot be created.
    */
    void addInputFile(const string &fileName);


    /**
    * @brief Sets the output directory for files.
    *
    * @param outputDir The output directory to be set.
    */
    void setOutputDirectory(const string &outputDir);

    /**
    * @brief Creates the output directory if it does not already exist.
    *
    * @throws runtime_error If the output directory cannot be created or opened.
    */
    void createOutputDirectory();
    
    /**
    * @brief Finalizes the parser by writing the meta file and cleaning up resources.
    * 
    */
    void finalize();

    /**
     * @brief Retrieves the number of rows in the current file context.
     *
     * @return The number of rows in the current file context.
     */
    int getCurrentFileNumRows() const;

    /**
     * @brief Retrieves the number of columns in the current file context.
     *
     * @return The number of columns in the current file context.
     */
    int getCurrentFileNumColumns() const;

    /**
     * @brief Retrieves the largest mapped value in the current file context.
     *
     * @return The largest mapped value in the current file context.
     */
    int getCurrentFileLargestMappedValue() const;

    /**
     * @brief Retrieves the number of bits needed for mapping in the current file context.
     *
     * @return The number of bits needed for mapping in the current file context.
     */
    int getCurrentFileNumBits() const;

    /**
     * @brief Retrieves the number of input files.
     *
     * @return The number of input files.
     */
    int getNumInputFiles() const;

private:

    // Private Members
    int totalNumRows;
    int totalNumColumns;
    string outputDirectory;
    vector<string> inputFileNames;
    FileContext CurrentFileContext;


    // Private Methods

    /**
    * @brief Ensures that the specified file ends with a newline character.
    *
    * This function opens the specified file and checks if the last character is a newline.
    * If the file does not end with a newline, it appends one to the file.
    *
    * @throws runtime_error If the file cannot be opened for reading or writing.
    */
    void ensureEndingNewLine();

    /**
    * @brief Retrieves the number of rows in the CSV file.
    *
    * This function opens the specified CSV file and reads through it to determine the number of rows.
    * It handles quoted fields to ensure that newline characters within quotes are not counted as row separators.
    * The function also records the file offsets of each row for later use.
    *
    * @return The number of rows in the CSV file.
    * @throws runtime_error If the file cannot be opened for reading.
    */
    int getRowCount();

    /**
    * @brief Retrieves the number of columns in the CSV file.
    *
    * This function opens the specified CSV file and reads the first line to determine the number of columns.
    * It handles quoted fields to ensure that commas within quotes are not counted as column separators.
    *
    * @return The number of columns in the CSV file.
    * @throws runtime_error If the file cannot be opened for reading.
    */
    int getColumnCount();

    /**
    * @brief Parses rows of a CSV file in parallel using multiple threads.
    *
    * This function divides the rows of a CSV file into groups and processes them in parallel using multiple threads.
    * Each thread processes a subset of rows and columns, mapping string values to integers.
    * The results are combined across threads to produce a final mapping for each column group.
    *
    * @throws runtime_error If any thread encounters an error during processing.
    */
    void parseRowParallel();

    /**
    * @brief Maps the transpose of the remapped values (aggregated over all threads for a column group) to a binary file.
    *
    * This function writes the remapped values for each column group to a binary file.
    * The values are stored in a compact binary format to save space.
    *
    * @param columnGroupContext A pointer to a ColumnGroupContext structure containing information about the column group.
    */
    void writeUpdatedMappedValues(ColumnGroupContext *columnGroupContext);
    void readAndDisplayCSV(ReadBinFileContext *readBinFileContext);

};

#endif // PARSER_H
