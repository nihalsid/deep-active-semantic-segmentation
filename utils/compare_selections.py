import os


def compare_selections(folder_a, folder_b):

    list_of_folders_a = [x for x in os.listdir(folder_a) if os.path.isdir(os.path.join(folder_a, x))]
    list_of_folders_b = [x for x in os.listdir(folder_b) if os.path.isdir(os.path.join(folder_b, x))]

    list_of_common_folders = sorted([x for x in list_of_folders_a if x in list_of_folders_b])

    last_selections_a = set()
    last_selections_b = set()

    for f in list_of_common_folders:
        selections_file_a = os.path.join(folder_a, f, 'selections.txt')
        selections_file_b = os.path.join(folder_b, f, 'selections.txt')
        with open(selections_file_a, "r") as fptr:
            selections_a = set(fptr.readlines()).difference(last_selections_a)
        with open(selections_file_b, "r") as fptr:
            selections_b = set(fptr.readlines()).difference(last_selections_b)
        last_selections_a = last_selections_a.union(selections_a)
        last_selections_b = last_selections_b.union(selections_b)
        assert len(selections_b) == len(selections_a), f"unequal number of selections in {f}"
        num_intersections = len(selections_a.intersection(selections_b))
        print(f'Number of common elements in {f} = {num_intersections}/{len(selections_a)} ({num_intersections * 100.0 / len(selections_a)})')


if __name__ == '__main__':
    import sys
    compare_selections(sys.argv[1], sys.argv[2])
