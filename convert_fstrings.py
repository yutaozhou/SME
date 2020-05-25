
from os.path import dirname, join
import future_fstrings
from glob import glob

def convert():
    files = join(dirname(__file__), "**/*.py")
    files = glob(files, recursive=True)
    for file in files:
        with open(file, "rb") as f:
            text, _ = future_fstrings.fstring_decode(f.read())
        with open(file, "w") as f:
            f.write(text)

if __name__ == "__main__":
    convert()
