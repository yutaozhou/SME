""" Version information """
from pathlib import Path

try:
    import git

    hasgit = True
except ImportError:
    hasgit = False

short_version = "0.2.0"
version = "0.2.0"
full_version = "0.2.0"

# Use current git information if possible
if hasgit:
    try:
        folder = Path(__file__).parent.parent.parent
        repo = git.Repo(folder)
        git_revision = str(repo.head.commit)
    except Exception:
        git_revision = "Unknown"
else:
    git_revision = "Unknown"
release = False

if not release:
    version = full_version + "." + git_revision
