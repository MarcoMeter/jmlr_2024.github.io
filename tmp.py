import os

# Path to the directory with your HTML files
directory = './results'

# The old function as a string
old_function = '''layout.width = window.innerWidth;'''

# The new function as a string
new_function = '''layout.width = window.innerWidth - 50;'''

# Iterate over all files in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    
    # Only process HTML files
    if filepath.endswith('.html'):
        print(filepath)
        with open(filepath, 'r') as file:
            filedata = file.read()
        
        # Replace the old function with the new function
        newdata = filedata.replace(old_function, new_function)
        
        # Write the changes back to the file
        with open(filepath, 'w') as file:
            file.write(newdata)
