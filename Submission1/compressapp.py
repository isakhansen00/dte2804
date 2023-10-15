import tkinter as tk
from tkinter import filedialog
from tkinter import ttk, messagebox
from os import path
import rle
import cv2 as cv
import os
import wave
import numpy as np
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel
import string
from dahuffman import HuffmanCodec
from lempelzivwelch import LZWCompress

def browse_file():
    file_path = filedialog.askopenfilename()
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def browse_destination():
    destination_path = filedialog.askdirectory()
    destination_entry.delete(0, tk.END)
    destination_entry.insert(0, destination_path)

def runlength(input_file, output_path):
    filename = path.basename(input_file)+"_compressed.txt"
    output_file = path.join(output_path, filename)

    # Read input file
    try:
        img_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif']
        sound_formats = ['.wav']
        file_extension = os.path.splitext(input_file)[1].lower() 
        if file_extension in img_formats:
            data = cv.imread(input_file).flatten()
        elif file_extension in sound_formats:
            with wave.open(input_file, 'rb') as audio_file:
                params = audio_file.getparams()
                data = np.frombuffer(audio_file.readframes(params.nframes), dtype=np.uint8)
        else:
            with open(input_file, 'r') as f:
                try:
                    data = f.read()
                except:
                    tk.messagebox.showwarning(title="Error", message="Error, the app can not compress this file type")
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' does not exist.")
        return

    # Encode the data using RLE
    encoded_data = rle.encode(data)

    # Save the encoded data to the output file
    try:
        if file_extension in img_formats:
            with open(output_file, 'w') as f:
                f.write(str(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
        elif file_extension in sound_formats:
            with open(output_file, 'w') as encoded_audio_file:
                encoded_audio_file.write(str(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
        else:
            with open(output_file, 'w') as f:
                f.write(str(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
    except Exception as e:
        print(e)

def arithmetic(input_file, output_path):

    numbers = {}
    numbers_string = {}
    alphabet = {}
    signs = {}
    hex_digits = {}
    whitespace = {}
    alphabet_list = list(string.ascii_letters)
    signs_list = list(string.punctuation) + ["\n"]
    hex_digits_list = list(string.hexdigits)
    whitespace_list = list(string.whitespace)

    # Making all the probabilities for characters
    for i in range(256):
        numbers[i] = 0.5
    
    for i in range(256):
        numbers_string[str(i)] = 0.5

    for i in alphabet_list:
        alphabet[i] = 0.5
    
    for i in signs_list:
        signs[i] = 0.5

    for i in hex_digits_list:
        hex_digits[i] = 0.5

    for i in whitespace_list:
        whitespace[i] = 0.5

    prob = {**numbers, **alphabet, **signs, **numbers_string, **hex_digits, **whitespace}
    
    # create the model
    model = StaticModel(prob)

    # create an arithmetic coder
    coder = AECompressor(model)

    filename = path.basename(input_file)+"_compressed.txt"
    output_file = path.join(output_path, filename)

    # Read input file
    try:
        img_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif']
        sound_formats = ['.wav']
        file_extension = os.path.splitext(input_file)[1].lower() 
        if file_extension in img_formats:
            data = cv.imread(input_file).flatten()
        elif file_extension in sound_formats:
            with wave.open(input_file, 'rb') as audio_file:
                params = audio_file.getparams()
                data = np.frombuffer(audio_file.readframes(params.nframes), dtype=np.uint8)
        else:
            with open(input_file, 'r') as f:
                try:
                    data = f.read()
                except:
                    tk.messagebox.showwarning(title="Error", message="Error, the app can not compress this file type")
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' does not exist.")
        return


    # Encode the data using Arithmetic
    encoded_data = coder.compress(data)

    # Save the encoded data to the output file
    try:
        if file_extension in img_formats:
            with open(output_file, 'w') as f:
                f.write(str(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
        elif file_extension in sound_formats:
            with open(output_file, 'w') as encoded_audio_file:
                encoded_audio_file.write(str(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
        else:
            with open(output_file, 'w') as f:
                f.write(str(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
    except Exception as e:
        print(e)

def huffman(input_file, output_path):

    img_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif']
    sound_formats = ['.wav']
    file_extension = os.path.splitext(input_file)[1].lower() 
    
    numbers = []
    numbers_string = []
    alphabet_list = list(string.ascii_letters)
    signs_list = list(string.punctuation) + ["\n"]
    whitespace_list = list(string.whitespace)

    freq_list = []

    for i in range(256):
        numbers.append(i)
    
    for i in range(256):
        numbers_string.append(str(i))

    if file_extension in img_formats or file_extension in sound_formats:
        freq_list = numbers
    else:
        freq_list = numbers_string + alphabet_list + signs_list + whitespace_list

    coder = HuffmanCodec.from_data(freq_list)

    filename = path.basename(input_file)+"_compressed.bin"
    output_file = path.join(output_path, filename)

    # Read input file
    try:
        if file_extension in img_formats:
            data = cv.imread(input_file).flatten()
        elif file_extension in sound_formats:
            with wave.open(input_file, 'rb') as audio_file:
                params = audio_file.getparams()
                data = np.frombuffer(audio_file.readframes(params.nframes), dtype=np.uint8)
        else:
            with open(input_file, 'r') as f:
                try:
                    data = f.read()
                except:
                    tk.messagebox.showwarning(title="Error", message="Error, the app can not compress this file type")
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' does not exist.")
        return


    # Encode the data using Arithmetic
    encoded_data = coder.encode(data)

    # Save the encoded data to the output file
    try:
        if file_extension in img_formats:
            with open(output_file, 'wb') as f:
                f.write(bytes(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
        elif file_extension in sound_formats:
            with open(output_file, 'wb') as encoded_audio_file:
                encoded_audio_file.write(bytes(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
        else:
            with open(output_file, 'w') as f:
                f.write(str(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
    except Exception as e:
        print(e)

def dictionary(input_file, output_path):
    filename = path.basename(input_file)+"_compressed.txt"
    output_file = path.join(output_path, filename)

    # Read input file
    try:
        img_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif']
        sound_formats = ['.wav']
        file_extension = os.path.splitext(input_file)[1].lower() 
        if file_extension in img_formats:
            data = cv.imread(input_file).flatten()
            data = data.tolist()
            data = map(str, data)
        elif file_extension in sound_formats:
            with wave.open(input_file, 'rb') as audio_file:
                params = audio_file.getparams()
                data = np.frombuffer(audio_file.readframes(params.nframes), dtype=np.uint8)
                data = data.tolist()
                data = map(str, data)
        else:
            with open(input_file, 'r') as f:
                try:
                    data = f.read()
                except:
                    tk.messagebox.showwarning(title="Error", message="Error, the app can not compress this file type")
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' does not exist.")
        return

    # Encode the data using RLE
    encoded_data = LZWCompress(data)

    # Save the encoded data to the output file
    try:
        if file_extension in img_formats:
            with open(output_file, 'w') as f:
                f.write(str(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
        elif file_extension in sound_formats:
            with open(output_file, 'w') as encoded_audio_file:
                encoded_audio_file.write(str(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
        else:
            with open(output_file, 'w') as f:
                f.write(str(encoded_data))
            tk.messagebox.showinfo(title="Compression Done", message="Compression Done")
    except Exception as e:
        print(e)

root = tk.Tk()
root.title("The Compressinator")
root.geometry("400x200")  # Set the window size

label = tk.Label(root, text="Select a File, Destination Folder, and Compression algorithm:")
label.pack()

file_frame = tk.Frame(root)
file_frame.pack()

file_label = tk.Label(file_frame, text="Select File:")
file_label.pack(side=tk.LEFT)

file_entry = tk.Entry(file_frame, width=40)
file_entry.pack(side=tk.LEFT)

file_browse_button = tk.Button(file_frame, text="Browse", command=browse_file)
file_browse_button.pack(side=tk.LEFT)

destination_frame = tk.Frame(root)
destination_frame.pack()

destination_label = tk.Label(destination_frame, text="Select Destination:")
destination_label.pack(side=tk.LEFT)

destination_entry = tk.Entry(destination_frame, width=40)
destination_entry.pack(side=tk.LEFT)

destination_browse_button = tk.Button(destination_frame, text="Browse", command=browse_destination)
destination_browse_button.pack(side=tk.LEFT)

option_label = tk.Label(root, text="Select Algorithm:")
option_label.pack()

options = ["RLE", "Arithmetic", "Huffman", "Dictionary-based"]
dropdown_var = tk.StringVar(root)
dropdown = ttk.Combobox(root, textvariable=dropdown_var, values=options)
dropdown.pack()

option_functions = {
    "RLE": runlength,
    "Arithmetic": arithmetic,
    "Hufman": huffman,
    "Dictionary-based": dictionary
}

def check_file_path():
    file_path = file_entry.get()
    file_destination = destination_entry.get()
    selected_algorithm = dropdown_var.get()

    if not path.exists(file_path):
        messagebox.showerror(title="Error", message="Sorry, it seems like the path to the file is not found. please specify a correct path")
    elif not path.exists(file_destination):
        messagebox.showerror(title="Error", message="Sorry, it seems like the path for the destination is incorrect. please try again.")
    elif selected_algorithm not in option_functions:
        messagebox.showerror(title="Error", message="Please choose an algorithm!")
    else:
        return True

def execute_selected_option():
    if check_file_path() == True:
        selected_option = dropdown_var.get()
        file_path = file_entry.get()
        file_destination = destination_entry.get()
        
        if selected_option in option_functions:
            option_functions[selected_option](file_path, file_destination)
        

button = tk.Button(root, text="Compress!", command=execute_selected_option)
button.pack()

root.mainloop()
