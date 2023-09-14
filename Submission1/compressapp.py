import tkinter as tk
from tkinter import filedialog
from tkinter import ttk, messagebox
from os import path
import rle

#TODO:
#Add error-handling
#Add success-message


def browse_file():
    file_path = filedialog.askopenfilename()
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def browse_destination():
    destination_path = filedialog.askdirectory()
    destination_entry.delete(0, tk.END)
    destination_entry.insert(0, destination_path)

def runlength(input_file, output_path): #doesn't work.
    filename = path.basename(input_file)+"_compressed.bin"
    output_file = path.join(output_path, filename)

    try:
        with open(input_file, 'rb') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' does not exist.")
        return

    # Check if the data is empty
    if not data:
        print(f"Warning: The input file '{input_file}' is empty. No compression performed.")
        return

    # Ensure that data is of type 'bytes'
    if not isinstance(data, bytes):
        print(f"Error: The input file '{input_file}' could not be read as bytes.")
        return

    # Encode the data using RLE
    encoded_data = rle.encode(data)

    # Save the encoded data to the output file
    with open(output_file, 'wb') as f:
        f.write(bytes(encoded_data))

    print(f"Encoded {input_file} to {output_file}")

def arithmetic():
    #insert algo here
    pass

def hufman():
    #insert algo here
    pass

def dictionary():
    #insert algo here
    pass

root = tk.Tk()
root.title("The Compressinator")
root.geometry("400x200")  # Set the window size

label = tk.Label(root, text="Select a File, Destination Folder, and compression algorithm:")
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

options = ["RLE", "Arithmetic", "Hufman", "Dictionary-based"]
dropdown_var = tk.StringVar(root)
dropdown = ttk.Combobox(root, textvariable=dropdown_var, values=options)
dropdown.pack()



option_functions = {
    "RLE": runlength,
    "Arithmetic": arithmetic,
    "Hufman": hufman,
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
