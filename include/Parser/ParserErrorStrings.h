#ifndef ParserErrorStrings_H
#define ParserErrorStrings_H

// Error Message Strings
#define ERR_ROW_EMPTY "Error: Every row must have at least one variable."
#define ERR_ROW_VAR_COUNT_MISMATCH "Error: Variable count for row %d does not match the first row's variable count of %d."
#define UNEXPECTED_TOKEN "Error: Unexpected token in file."
#define FAILED_TO_CREATE_OPEN_DATA_DIR "Failed to create or open data directory."
#define ERR_FILE_READ "Error: Failed to open file for reading."
#define FAIL_OPEN_BIN_FILE "Failed to open binary file for writing."
#define BITWIDTH_NOT_SET "Error: numBits not set. Ensure you calculate numBits before writing binary files."
#define ERR_FILE_EMPTY "Error: File is empty."
#define ERR_FILE_ROW_MISMATCH "Error: Number of rows in the file does not match the previous file."

#endif // ParserErrorStrings_H