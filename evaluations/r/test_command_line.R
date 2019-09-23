#!/usr/bin/env Rscript

## Usage
## Make script executable, e.g. >> chmod 777 test_command_line.R
## Run the script with the arguments >> ./test_command_line.R  Nikhil Monday


# Read the arguments
args = commandArgs(trailingOnly=TRUE)

# Access as a list
loName = args[1]
loDay = args[2]

print(paste0("Name: ", loName, ", Day: ", loDay))
    
