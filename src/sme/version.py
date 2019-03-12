""" Version information """
try:
    import git
    import os

    hasgit = True
except ImportError:
    hasgit = False

short_version = "0.2.0"
version = "0.2.0"
full_version = "0.2.0"

# Use current git information if possible
if hasgit:
    try:
        repo = git.Repo(os.path.dirname(__file__) + "/../..")
        git_revision = str(repo.head.commit)
    except Exception:
        git_revision = "Unknown"
else:
    git_revision = "Unknown"
release = False

if not release:
    version = full_version + "." + git_revision
