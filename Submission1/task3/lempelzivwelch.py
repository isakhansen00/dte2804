import string

def LZWCompress(uncompressed):
    """Compress a string to a list of output symbols."""

    prob = {}
    dict_size = 256

    for i in range(256):
        prob[chr(i)] = i

    w = ""
    result = []
    for string_to_compress in uncompressed:
        for c in string_to_compress:
            wc = w + c
            if wc in prob:
                w = wc
            else:
                result.append(prob[w])
                # Add wc to the dictionary.
                prob[wc] = dict_size
                dict_size += 1
                w = c

    # Output the code for w.
    if w:
        result.append(prob[w])
    return result
def main():
    compressed = LZWCompress(["255", "255", "255", "255"])
    print(compressed)

if __name__ == "__main__":
    main()
