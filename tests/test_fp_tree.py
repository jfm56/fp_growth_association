from src.fp_tree import FPTree

def test_fp_tree_insert_and_visualize(tmp_path):
    tree = FPTree()
    transactions = [
        ['a', 'b', 'c'],
        ['a', 'b'],
        ['b', 'c'],
        ['a', 'c'],
        ['b', 'c']
    ]
    for tr in transactions:
        tree.insert_transaction(tr)
    # Should have root and children
    assert tree.root is not None
    assert len(tree.root.children) > 0
    # Visualize (output file should be created)
    output_file = tmp_path / "fp_tree_test.gv"
    tree.visualize(str(output_file))
    assert output_file.with_suffix('.gv.pdf').exists() or output_file.with_suffix('.gv.png').exists()
