import json
import re

def convert_script_to_notebook(script_path, output_path):
    with open(script_path, 'r') as f:
        content = f.read()
    
    cells = []
    current_cell = []
    current_cell_type = "code"
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('# @title '):
            if current_cell:
                cell_content = '\n'.join(current_cell).strip()
                if cell_content:
                    cells.append({
                        "cell_type": current_cell_type,
                        "metadata": {},
                        "source": [cell_content + '\n'] if cell_content else [],
                        "outputs": [] if current_cell_type == "code" else None,
                        "execution_count": None if current_cell_type == "code" else None
                    })
            
            current_cell = [line]
            current_cell_type = "code"
        else:
            current_cell.append(line)
        
        i += 1
    
    if current_cell:
        cell_content = '\n'.join(current_cell).strip()
        if cell_content:
            cells.append({
                "cell_type": current_cell_type,
                "metadata": {},
                "source": [cell_content + '\n'] if cell_content else [],
                "outputs": [] if current_cell_type == "code" else None,
                "execution_count": None if current_cell_type == "code" else None
            })
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    convert_script_to_notebook("gencast_mini_demo.txt", "gencast_mini_demo_fixed.ipynb")
    print("Converted script to notebook successfully!")
