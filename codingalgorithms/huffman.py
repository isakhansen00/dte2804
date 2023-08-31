
from collections import Counter


class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def child_nodes(self):
        return self.left, self.right

    def __str__(self):
        return self.left, self.right


def huffman_tree(node, binary_string=''):
    '''
    Function to find Huffman Code
    '''
    if type(node) is str:
        return {node: binary_string}
    (l, r) = node.child_nodes()
    d = dict()
    d.update(huffman_tree(l, binary_string + '0'))
    d.update(huffman_tree(r, binary_string + '1'))
    return d


def make_tree(nodes):
    '''
    Function to make tree
    :param nodes: Nodes
    :return: Root of the tree
    '''
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]


if __name__ == '__main__':
    string = 'isak'
    freq = dict(Counter(string))
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    node = make_tree(freq)
    encoding = huffman_tree(node)
    print(encoding)