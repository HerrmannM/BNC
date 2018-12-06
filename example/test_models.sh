#!/bin/bash
# Dr Matthieu Herrmann, Monash University, Melbourne, Australie
# 2018
# GPL v3

### ### ### Coding environment
# e = exit on error, u = unset variable as error, -o pipefail: check all command from a pipeline
# x = print command before execution
set -euo pipefail
# set -x

### ### ### CONSTANTS

# Get the script absolute path and separate in path and name
SCRIPT=$(realpath $0)
SCRIPT_PATH=$(dirname $SCRIPT)
SCRIPT_NAME=$(basename $SCRIPT)

# We use the same evaluator for all experiments.
# Warning: we need to add the file path at the end of this command.
JAVACMD="java -Xms1g -Xmx1g -jar $SCRIPT_PATH/../dist/bnc.jar --evaluator kfoldxval 2 5 "

# Declare the configuration for the different model
declare -a arr=(
"SKDB -I 1000 -K 5 -L 2"
"SKDB -I 1000 -K 5 -L 2 -M"
"ESKDB -I 1000 -K 5 -L 2 -E 5"
"ESKDB -I 1000 -K 5 -L 2 -M -E 20")

# Folder to store the results in
RESULT_FOLDER="result/"

# Folder containing the data
DATA_FOLDER="data"

# Temporary file
TMP_FILE=$(mktemp $SCRIPT_NAME.XXXXXX)
echo "Temporary file $TMP_FILE created"


### ### #### Do the work

# Create the result folder
mkdir -p "$RESULT_FOLDER"

## now loop through the model parameter. Create a result file per model.
for modelParam in "${arr[@]}"
do
  modelParam_="${modelParam// /_}.csv"
  outFile="$RESULT_FOLDER$modelParam_"
  echo "create/truncate the file $outFile"
  truncate -s 0 "$outFile"

  # For each file in $DATA_FOLDER
  while read file
  do
    # Create the command line and execute it
    cmd="$JAVACMD $file --model $modelParam"
    echo $cmd
    cmd="$cmd | tail -n15 >$TMP_FILE"
    eval "$cmd"

    # Extract the info
    fileName=$(sed -n 1p "$TMP_FILE" | tr -s ' ' | cut -d' ' -f2)
    fileName=$(basename $fileName .arff)
    RMSE=$(sed -n 13p "$TMP_FILE" | tr -s ' ' | cut -d' ' -f2)
    ERROR=$(sed -n 14p "$TMP_FILE" | tr -s ' ' | cut -d' ' -f2)
    CSVLINE="$fileName, $RMSE, $ERROR"
    echo "$CSVLINE"
    echo "$CSVLINE" >> "$outFile"

  done < <(find "$DATA_FOLDER" -type f -name "*.arff" ;)

done

# remove temporary file
rm "$TMP_FILE"
