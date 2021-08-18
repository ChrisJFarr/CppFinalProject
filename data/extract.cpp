#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;
// Data from https://www.kaggle.com/c/digit-recognizer/data


// Create functions to perform when iterating through examples
// Create an iterator function that accepts another function and iterates...
int countRows(string filePath, bool header=true)
{
    // Open file
    ifstream file("train.csv");  //, ios::in
    int nRows = 0;
    bool firstRow = true;  // Intialize this with true, set to false after first line is parsed
    string line, dataPoint;
    vector<float> row;
    if(file.is_open())
    {
        while(file >> line)
        {
            // If header, skip the first line
            if(header && firstRow){firstRow=false;continue;};
            // Increment nRows while looping through file
            nRows++;
            row.clear();
            // Create a stream object
            stringstream s(line);
            while (getline(s, dataPoint, ',')) {
                row.push_back(stof(dataPoint));
            }
        }
    }
    return nRows;
}


int main()
{

    // TODO's

    // Create function that loads a file and reads how many examples exist
    string filePath = "train.csv";
    cout << "Number of examples:" << countRows(filePath) << endl;




    // Extract data to Matrix objects once finished implementing
    // Extract data to vector for now for practice
    ifstream file("train.csv");  //, ios::in

    // How many rows? Its a new line when a comma has no trailing data...
    // Label is in the first row
    vector<float> row;
    string line, word;

    if(file.is_open())
    {
        cout << "file opened" << endl;
        // Read a few hundred places and send to stdout
        int i = 0;
        while(file >> line)
        {
            cout << "printing line:";
            for(auto r: line){cout << r;}
            cout << endl;  

            // cout << "printing temp:";
            // for(auto r: temp){cout << r;}
            // cout << endl;  

            // Skipping first row
            if(i==0){i++;continue;}
            
            row.clear();

            // used for breaking words
            stringstream s(line);
    
            // read every column data of a row and
            // store it in a string variable, 'word'
            while (getline(s, word, ',')) {
                row.push_back(stof(word));
            }
            // for(auto r: row)
            // {
            //     cout << r;
            // }
            // cout << endl;  
            break;
        }
    } 


// TODO's (measure the time for each activity)
// Count the number of examples in the file
// Read the last 50 examples and store in a unique_prt<vector<vector<int>>



return 0;
}
