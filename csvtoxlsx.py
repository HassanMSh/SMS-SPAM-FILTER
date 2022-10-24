# install the following libraries using pip
import pandas as pd
import os

# Place the path to the folder containing the csv files here
path = 'C:/Users/bmt09/Desktop/FYP/Test Project 1/Datasets'

# to specified directory
os.chdir(path)

# reads the csv file


def read_text_file(path, file):
    # Dictionary to store the data
    DataFrames = {}

    # add the read file to the dataframe
    file_path = f"{path}\{file}"
    df = pd.read_csv(file_path,
                     sep=';', header=1, on_bad_lines='skip')

    # DataFrameNumber is the key (to iterate through the dictionary) and the DataFrame is the value
    DataFrameNumber = 1

    MaxRows = 999999

    # iterate through the DataFrame
    # len(df) is the number of rows in the DataFrame
    for i in range(0, len(df), MaxRows):
        print("Success")
        # if the number of rows is less than the max number of rows
        # add the DataFrame to the dictionary
        if (i + MaxRows > len(df)):
            DataFrames[DataFrameNumber] = df[i:]
        # if the number of rows is greater than the max number of rows
        # export csv file to the specified directory in .xlsx format
        else:
            DataFrames[DataFrameNumber] = df[i:i+MaxRows]
        DataFrames[DataFrameNumber].to_excel(path + "/Output/" + os.path.splitext(
            file)[0] + "_" + str(DataFrameNumber) + '.xlsx', engine='xlsxwriter', merge_cells=False, index=False)
        # increment the key
        DataFrameNumber += 1


# create a new folder to store the output files
try:
    os.mkdir(path+"/Output")
except OSError as error:
    print(error)

# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".csv"):
        # call read text file function
        read_text_file(path, file)
