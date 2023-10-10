#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <queue>
using namespace std;

// to map each character to its Huffman value
map<char, string> codes;

// To store the frequency of characters in the input data
map<char, int> freq;

// A Huffman tree node
struct MinHeapNode {
    char data; // One of the input characters
    int freq; // Frequency of the character
    MinHeapNode* left, * right; // Left and right child

    MinHeapNode(char data, int freq) {
        left = right = nullptr;
        this->data = data;
        this->freq = freq;
    }
};

// utility function for the priority queue
struct compare {
    bool operator()(MinHeapNode* l, MinHeapNode* r) {
        return (l->freq > r->freq);
    }
};

// utility function to print characters along with their Huffman value
void printCodes(struct MinHeapNode* root, string str) {
    if (!root)
        return;
    if (root->data != '$')
        cout << root->data << ": " << str << "\n";
    printCodes(root->left, str + "0");
    printCodes(root->right, str + "1");
}

// utility function to store characters along with their Huffman value in a hash table
void storeCodes(struct MinHeapNode* root, string str) {
    if (root == nullptr)
        return;
    if (root->data != '$')
        codes[root->data] = str;
    storeCodes(root->left, str + "0");
    storeCodes(root->right, str + "1");
}

// STL priority queue to store heap tree nodes
priority_queue<MinHeapNode*, vector<MinHeapNode*>, compare> minHeap;

// function to build the Huffman tree and store it in minHeap
void HuffmanCodes(int size) {
    struct MinHeapNode* left, * right, * top;
    for (map<char, int>::iterator v = freq.begin(); v != freq.end(); v++)
        minHeap.push(new MinHeapNode(v->first, v->second));
    while (minHeap.size() != 1) {
        left = minHeap.top();
        minHeap.pop();
        right = minHeap.top();
        minHeap.pop();
        top = new MinHeapNode('$', left->freq + right->freq);
        top->left = left;
        top->right = right;
        minHeap.push(top);
    }
    storeCodes(minHeap.top(), "");
}

// utility function to store map each character with its frequency in input string
void calcFreq(string str, int n) {
    for (int i = 0; i < str.size(); i++)
        freq[str[i]]++;
}

// Function to get the size of a file 
long long getFileSize(const string& fileName) {
    ifstream file(fileName, ios::binary | ios::ate);
    if (file.is_open()) {
        long long fileSize = file.tellg();
        file.close();
        return fileSize;
    }
    else {
        return -1; // Error: File not found or couldn't be opened
    }
}

// Function for writing the huffman tree to file as binary format
void encodeToFile(const string& inputFileName, const string& outputFileName) {
    ifstream inputFile(inputFileName);
    string inputString((istreambuf_iterator<char>(inputFile)), (istreambuf_iterator<char>()));
    calcFreq(inputString, inputString.length());
    HuffmanCodes(inputString.length());

    ofstream outputFile(outputFileName, ios::binary);
    if (!outputFile.is_open()) {
        cout << "Error opening output file." << endl;
        return;
    }

    string encodedBits = "";
    for (auto i : inputString)
        encodedBits += codes[i];

    // Convert binary string to actual binary data
    vector<unsigned char> binaryData;
    for (size_t i = 0; i < encodedBits.size(); i += 8) {
        string byteStr = encodedBits.substr(i, 8);
        unsigned char byte = static_cast<unsigned char>(stoi(byteStr, nullptr, 2));
        binaryData.push_back(byte);
    }

    // Write binary data to the output file
    outputFile.write(reinterpret_cast<const char*>(binaryData.data()), binaryData.size());

    inputFile.close();
    outputFile.close();

    cout << "Encoding complete. Output saved to " << outputFileName << endl;
}



int main() {
    string inputFileName = "input.txt";
    string encodedFileName = "encoded.bin";


    // Encode the input file
    encodeToFile(inputFileName, encodedFileName);

    // Get the file size of input.txt
    long long inputFileSize = getFileSize(inputFileName);
    if (inputFileSize != -1) {
        cout << "Size of input.txt: " << inputFileSize << " bytes" << endl;
    }
    else {
        cout << "Error: input.txt not found or couldn't be opened." << endl;
    }

    // Get the file size of the compressed file (encoded.bin)
    long long encodedFileSize = getFileSize(encodedFileName);
    if (encodedFileSize != -1) {
        cout << "Size of encoded.bin: " << encodedFileSize << " bytes" << endl;
    }
    else {
        cout << "Error: encoded.bin not found or couldn't be opened." << endl;
    }

    return 0;
}





