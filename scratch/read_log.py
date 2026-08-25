import sys
import os

def read_last_lines(filepath, num_lines=150):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} does not exist.")
        return
    with open(filepath, 'rb') as f:
        # Go to the end of the file
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        
        # Read backwards to find num_lines newlines
        buffer_size = 1024
        lines_found = 0
        pos = file_size
        
        while pos > 0 and lines_found <= num_lines:
            pos = max(0, pos - buffer_size)
            f.seek(pos)
            chunk = f.read(min(buffer_size, file_size - pos))
            lines_found += chunk.count(b'\n')
            
        # Now seek to the estimated start position
        f.seek(pos)
        content = f.read()
        lines = content.split(b'\n')
        # Print the last num_lines lines
        for line in lines[-num_lines:]:
            print(line.decode('utf-8', errors='replace'))

if len(sys.argv) < 2:
    print("Usage: python read_log.py <filepath> [num_lines]")
else:
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    read_last_lines(path, n)
