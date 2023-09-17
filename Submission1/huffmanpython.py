# https://narainsreehith.medium.com/huffman-coding-on-image-d6092bed5821

import cv2
import numpy as np
from copy import deepcopy as cp
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import filedialog

"""
The code aims to apply Huffman coding to compress an image by representing frequently occurring intensity values with shorter binary codes and less frequent values with longer codes. 
The Image class provides methods for encoding and decoding the image using Huffman coding.
This code defines two classes, Node and Image, for implementing the Huffman coding algorithm on an image.
"""

# This class represents a node in the Huffman coding tree.
class Node:
    def __init__(self, f, p, isLeaf=0):
        self.freq = f          # For storing intensity
        self.prob = p          # For storing Probability
        self.word = ""         # For coded word ex : '100101'
        self.c = [None, None]  # Pointers for children, C[0] for left child, c[1] for right child
        self.isLeaf = isLeaf   # Flag for Leaf-Nodes

# This class represents an image and contains methods to perform Huffman coding and decoding on the image.
class Image:
    def __init__(self):
        self.path_in = ""               # Input file Location
        self.path_out = ""              # Output file Location
        self.im = np.zeros(1)           # Input Image
        self.out = np.zeros(1)          # Output Image
        self.image_data = np.zeros(1)   # List of Intensities and dimensions
        self.r = 0                      # Rows
        self.c = 0                      # Coloums
        self.d = 0                      # Depth / channels

        # histogram, ie frequency count of each data value
        self.hist = np.zeros(1)
        self.freqs = np.zeros(1)        # For Non-zero frequency

        # Dictionary with key as freqs and values as Probailities
        self.prob_dict = {}
        self.allNodes = []              # Container for  All created Nodes
        self.leafNodes = {}             # Container for Leaf Nodes
        self.root = Node(-1, -1)        # Root Node with Probability = 1

        # Encoded String of Image,form: "01001010101011......", interpretation :  [r,c,d,[..pxls]]
        self.encodedString = ""

        # Decoded List of integers when read from .bin file, form : [456,342,3,34,2,120,44, ...... ], interpretation : [r,c,d,[..pxls]]
        self.decodeList = []

        # Binary from file in for of integers ie [0,1,0,0,1,0,1,0,1,0,1,0,1,1,......]
        self.binaryFromFile = []

    # Checks if the Huffman coding was successful by comparing the original image (im) and the decoded image (out).
    def checkCoding(self):
        return np.all(self.im == self.out)

    # Reads an image from the specified path and stores it in the im attribute using OpenCV (cv2).
    def readImage(self, path):
        self.path_in = path
        try:
            self.im = cv2.imread(path)
        except:
            print("Error in reading image")

    # Initializes the image data by flattening the image, adding its dimensions, and creating a histogram.
    def initialise(self):
        self.r, self.c, self.d = self.im.shape

        # Pushing r,c,d to encode into image_data list
        temp_list = self.im.flatten()
        temp_list = np.append(temp_list, self.r)
        temp_list = np.append(temp_list, self.c)
        temp_list = np.append(temp_list, self.d)

        self.image_data = temp_list

        # Creating historgram from image_data to create frequencies.
        self.hist = np.bincount(
            self.image_data, minlength=max(256, self.r, self.c, self.d))
        total = np.sum(self.hist)

        # Extracting the non-zero frequencies
        self.freqs = [i for i, e in enumerate(self.hist) if e != 0]
        self.freqs = np.array(self.freqs)

        # Creatn=ing a dict of propabilities , with keys are intensities and value as propabilities
        for i, e in enumerate(self.freqs):
            self.prob_dict[e] = self.hist[e]/total

    # Writes the encoded image to the specified output path using OpenCV.
    def outImage(self, path):
        self.path_out = path
        try:
            cv2.imwrite(self.path_out, self.out)
        except:
            print("Error in writing the image")

    # Creates nodes for each intensity value based on their probabilities.
    def buildNodes(self):
        for key in self.prob_dict:
            leaf = Node(key, self.prob_dict[key], 1)
            self.allNodes.append(leaf)

    # A comparison function used for sorting nodes by their probabilities.
    def prob_key(self, e):
        return e.prob

    # Constructs the Huffman tree (upward) by merging nodes with the lowest probabilities until a single root node is created.
    def upTree(self):

        import heapq
        self.buildNodes() # Creating Nodes

        # Sorting all Nodes in workspace to create uptree
        workspace = sorted(cp(self.allNodes), key=self.prob_key)
        while(1):
            c1 = workspace[0]
            c2 = workspace[1]
            workspace.pop(0)
            workspace.pop(0)

            # Creating A new node from  two smallest propability intensities
            new_node = Node(-1, c1.prob+c2.prob)
            new_node.c[0] = c1
            new_node.c[1] = c2

            workspace = list(heapq.merge(
                workspace, [new_node], key=self.prob_key)) # Pushing the created Node into Workspace
            # Break if probability of prepared node is 1, indicating preparing upTree is completed
            if(new_node.prob == 1.0):
                self.root = new_node        # And storing it as root Node
                return

    # Assigns Huffman codes (words) to leaf nodes (downward) starting from the root node.
    def downTree(self, root, word):
        root.word = word
        if(root.isLeaf):
            self.leafNodes[root.freq] = root.word
        if(root.c[0] != None):
            self.downTree(root.c[0], word+'0')
        if(root.c[1] != None):
            self.downTree(root.c[1], word+'1')

    # Combine the upTree and downTree functions to perform Huffman coding on the image, resulting in an encoded string.
    def huffmanAlgo(self):
        self.upTree()                   # Creating UpTree
        self.downTree(self.root, "")    # Creating DownTree

        dicti = {}                          # Storing the prob_dict in new variable dicti
        # So that we need not access ("self.") every time that costs time, we just use dicti in place of self.leafNodes
        for key in self.leafNodes:
            dicti[key] = self.leafNodes[key]

        # Storing the self.encodedString in new variable encodedString
        # So that we need not accecess "self." every time,which cost more time
        encodedString = ""
        encodedString += dicti[self.r]
        encodedString += dicti[self.c]
        encodedString += dicti[self.d]

        # Note we are first encoding dimensions, and later encoding each pxl in 3rd dimension order , later while decoding we decode in the same way

        for i in range(self.r):
            for j in range(self.c):
                for ch in range(self.d):
                    encodedString += dicti[self.im[i][j][ch]]

        self.encodedString = encodedString

    # Writes the encoded string to a binary file using the bitstring library.
    def sendBinaryData(self, path):
        #  Our self.encodedString is just list of strigs with characters char('0') & char('1')
        # but we should not directly write char('0') and char('1') , because each of them take 1byte = 8bits, so we are converting the char('0') and char('1') to binary(0) & binary(1)
        # To do the above work we are using bitstring from BitArray library
        from bitstring import BitArray
        file = open(path, 'wb')
        obj = BitArray(bin=self.encodedString)
        obj.tofile(file)
        file.close()

    # Reads the encoded binary data from a file, decodes it, and stores the decoded values in decodeList.
    def decode(self, path):
        # Now we have a file , we will try to read its contents
        # For reading binary from file we are using bitarray library (this is different one from above used BitArray)

        import bitarray
        self.binaryFromFile = bitarray.bitarray()
        with open(path, 'rb') as f:
            self.binaryFromFile.fromfile(f)

        # Data type of self.binaryFromFile is list of int(0)'s and int(1)'s

        decodeList = []
        root = self.root
        temp_root = cp(self.root)

        temp_r = 0
        temp_c = 0
        temp_d = 0

        # Like we encoded r,c,d, we will first retrieve/decode them

        for i, c_int in enumerate(self.binaryFromFile):
            if(temp_r != 0 and temp_c != 0 and temp_d != 0 and len(decodeList) == (temp_r*temp_c*temp_d + 3)):
                break
            if(temp_r == 0 and len(decodeList) >= 1):
                temp_r = decodeList[0]

            if(temp_c == 0 and len(decodeList) >= 2):
                temp_c = decodeList[1]

            if(temp_d == 0 and len(decodeList) >= 3):
                temp_d = decodeList[2]

            # Start at root, if 0 go to left node, if 1 go to right node, if leafnode is found store the intensity in a list
            # and follow the same procedure untill you find (r*c*d + 3) values, ie 3 dimensions and r*c*d intensities

            temp_root = temp_root.c[c_int]
            if(temp_root.isLeaf):
                decodeList.append(temp_root.freq)
                temp_root = root
                continue

        self.decodeList = decodeList

    # Reconstructs the image from the decoded values, considering dimensions and intensity values.
    def decodeIm(self, path):
        self.decode(path)

        # Extracting dimensions values from deocded list, ie its first 3 three

        decodeList = self.decodeList
        out_r = decodeList[0]
        decodeList.pop(0)
        out_c = decodeList[0]
        decodeList.pop(0)
        out_d = decodeList[0]
        decodeList.pop(0)

        out = np.zeros((out_r, out_c, out_d))

        # Filling the output image

        for i in range(len(decodeList)):
            id = i//out_d
            x = id//out_c
            y = id % out_c      # Basic parsing mathematics for finding index values
            z = i % out_d
            out[x][y][z] = decodeList[i]
        out = out.astype(dtype=int)
        self.out = out

    def huffmanCode(self, input_pth, compressed_pth="./compressed.bin", output_pth="./output.png", toCheck=0):
        # Just Steps for Execution

        import time

        self.readImage(input_pth)
        self.initialise()
        print('Initialization Done\n')

        s = time.time()
        print('Coding Image started\n')
        self.huffmanAlgo()
        print('Coding Image completed\n')
        e = time.time()
        encode_time = e-s

        print("\nOriginal Size of Image : ", self.r*self.c*self.d*8, " bits")
        print("\nCompressed Size : ", len(self.encodedString), " bits")
        print("\nCompressed factor : ", self.r *
              self.c*self.d*8/(len(self.encodedString)), "\n")

        print("Took ", encode_time, " sec to encode input image\n")
        print('Sending coded data\n')
        self.sendBinaryData(compressed_pth)
        print('Soded data sent\n')

        s = time.time()

        print('Started decoding compressed image\n')
        self.decodeIm(compressed_pth)
        self.outImage(output_pth)
        print('Completed decoding compressed image ( open output image from the above mentioned path) \n')

        e = time.time()
        decode_time = e-s
        print("Took ", decode_time, " sec to decode compressed image\n")
        if(toCheck):
            print("Are both images same : ", self.checkCoding())

        org_size_of_image = self.r*self.c*self.d*8
        comp_size_image = len(self.encodedString)
        comp_factor = self.r * self.c*self.d*8/(len(self.encodedString))
        

        # Ploting same results over in pyplot for nicer view
        # create figure
        fig = plt.figure(figsize=(12, 9))
        
        # setting values to rows and column variables
        rows = 2
        columns = 2
        
        # reading images
        org_img = mpimg.imread(input_pth)
        decompressed_img = mpimg.imread(output_pth)
        
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 1)
        
        # showing original image
        plt.imshow(org_img)
        plt.axis('off')
        plt.title("Original image", weight="bold")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
        
        # showing decompressed image
        plt.imshow(decompressed_img)
        plt.axis('off')
        plt.title("Decompressed image", weight="bold")

        # showing size of original image in bits
        fig.add_subplot(rows, columns, 3)
        plt.axis('off')
        plt.text(0.22, 0.9, 'Original Size of Image :', weight="bold", fontsize = 18)

        fig.add_subplot(rows, columns, 4)
        plt.axis('off')
        plt.text(-0.1, 0.9, str(org_size_of_image) + " bits", fontsize = 18)

        # showing compression size in bits
        fig.add_subplot(rows, columns, 3)
        plt.axis('off')
        plt.text(0.22, 0.75, 'Compressed Size :', weight="bold", fontsize = 18)

        fig.add_subplot(rows, columns, 4)
        plt.axis('off')
        plt.text(-0.1, 0.75, str(comp_size_image) + " bits", fontsize = 18)

        # showing compression factor
        fig.add_subplot(rows, columns, 3)
        plt.axis('off')
        plt.text(0.22, 0.6, 'Compressed factor :', weight="bold", fontsize = 18)

        fig.add_subplot(rows, columns, 4)
        plt.axis('off')
        plt.text(-0.1, 0.6, str(comp_factor), fontsize = 18)

        # showing time used for encoding image
        fig.add_subplot(rows, columns, 3)
        plt.axis('off')
        plt.text(0.22, 0.45, 'Time encoding :', weight="bold", fontsize = 18)

        fig.add_subplot(rows, columns, 4)
        plt.axis('off')
        plt.text(-0.1, 0.45, str(encode_time) + " seconds", fontsize = 18)

        # showing time used for decoding image
        fig.add_subplot(rows, columns, 3)
        plt.axis('off')
        plt.text(0.22, 0.3, 'Time decoding :', weight="bold", fontsize = 18)

        fig.add_subplot(rows, columns, 4)
        plt.axis('off')
        plt.text(-0.1, 0.3, str(decode_time) + " seconds", fontsize = 18)

        # showing time used for decoding image
        fig.add_subplot(rows, columns, 3)
        plt.axis('off')
        plt.text(0.25, 2.4, 'Huffman encoding of image using Python', weight="bold", fontsize = 22)

        plt.show()

class ImageHuffmanEncodingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Huffman Encoder")
        self.root.geometry("400x100")
        self.file_path = None

        label = tk.Label(root, text="Select a PNG/JPEG file and encode with huffman:")
        label.pack()

        file_frame = tk.Frame(root)
        file_frame.pack()

        file_label = tk.Label(file_frame, text="Select File:")
        file_label.pack(side=tk.LEFT)

        self.file_entry = tk.Entry(file_frame, width=40)
        self.file_entry.pack(side=tk.LEFT)

        # Create buttons
        self.select_button = tk.Button(file_frame, text="Select Image", command=self.select_image)
        self.select_button.pack()

        self.generate_button = tk.Button(root, text="Encode Image", command=self.encode_image)
        self.generate_button.pack()

    def select_image(self):
        self.file_path = filedialog.askopenfilename(title="Select an image")
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, self.file_path)
    
    def encode_image(self):
        if self.file_path:
            image = Image()
            image.huffmanCode(self.file_path, "./compressed.bin",
                            "./output.png", toCheck=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageHuffmanEncodingApp(root)
    root.mainloop()

"""
Filstørrelsen til compressed.bin blir større enn det originale sui.png-bildet fordi Huffman-kodealgoritmen implementert i koden din ikke er laget for å redusere størrelsen på bildet. 
Huffman-koding er en prefikskodingsalgoritme med variabel lengde som brukes for tapsfri datakomprimering, og effektiviteten avhenger av fordelingen av symboler (i dette tilfellet pikselintensiteter).
I koden brukes Huffman-koding for å kode pikselintensiteten til bildet. Redundans eller mønstre i bildedataene blir ikke utnyttet, noe som er avgjørende for å oppnå komprimering. 
Huffman-koding kan til og med øke størrelsen på dataene i noen tilfeller hvis symbolene har like sannsynligheter eller hvis det ikke er noen redundans å utnytte.
For å oppnå faktisk bildekomprimering, vil du vanligvis bruke bildespesifikke komprimeringsalgoritmer som JPEG, PNG eller andre som er designet for å utnytte romlig redundans og andre egenskaper ved bilder. 
Disse algoritmene bruker teknikker som diskret cosinustransformasjon (DCT), run-length-koding (RLE) og mer for å redusere filstørrelsen samtidig som den essensielle visuelle informasjonen til bildet bevares.
"""