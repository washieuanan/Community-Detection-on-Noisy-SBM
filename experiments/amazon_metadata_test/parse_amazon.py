import re

def parse_amazon_meta(path):
    """
    Parse the SNAP amazon-meta.txt file into a dict:
      ASIN → {
        'group': str,
        'similar': [ASIN, …],
        'categories': [label, …]
      }
    """
    products = {}
    with open(path, 'r', encoding='utf-8') as f:
        asin = None
        group = None
        similars = []
        categories = []
        cat_lines_to_read = 0

        def save_record():
            if asin is not None:
                products[asin] = {
                    'group':     group,
                    'similar':   similars,
                    'categories': categories
                }

        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith('Id:'):
                # new block: save previous
                save_record()
                # reset
                asin = group = None
                similars = []
                categories = []
                cat_lines_to_read = 0

            elif line.startswith('ASIN:'):
                asin = line.split('ASIN:')[1].strip()

            elif line.startswith('group:'):
                group = line.split('group:')[1].strip()

            elif line.startswith('similar:'):
                parts = line.split()
                # format: similar: <count>  ASIN1  ASIN2  …
                if len(parts) >= 3:
                    similars = parts[2:]

            elif line.startswith('categories:'):
                # next N lines are category‐paths
                parts = line.split()
                try:
                    cat_lines_to_read = int(parts[1])
                except ValueError:
                    cat_lines_to_read = 0

            elif cat_lines_to_read > 0:
                tokens = re.findall(r'\|([^|\[]+)\[\d+\]', line)
                if tokens:
                    categories.append(tokens[-1])
                cat_lines_to_read -= 1

            # ignore everything else (title, salesrank, reviews, etc.)

        # end‐of‐file: save last
        save_record()

    return products

if __name__ == '__main__':
    import json
    import os  
    
    products = parse_amazon_meta('amazon_metadata_test/amazon-meta.txt')
    with open('amazon_metadata_test/parsed_amazon_meta.json', 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2)