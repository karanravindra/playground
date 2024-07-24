import os

files = os.listdir(path="/Users/karan/projects/playground/misc/stories/docs")

count = 0
for file in files:
    with open(f"/Users/karan/projects/playground/misc/stories/docs/{file}") as f:
        count += len(f.read())

print(f"Total Characters in 'docs': {count:,}")
