import subprocess
import os
from datetime import datetime

def get_largest_git_objects(limit=20):
    print("📦 Scanning Git history for large files...")

    cmd = '''
    git rev-list --objects --all |
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize:disk) %(rest)' |
    grep '^blob' |
    sort -k3 -n -r |
    head -{}
    '''.format(limit)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    files = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 4:
            size = int(parts[2])
            filepath = " ".join(parts[3:])
            files.append((size, filepath))
    return files

def human_readable_size(size):
    for unit in ['B','KB','MB','GB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"

def prompt_files_to_delete(files):
    print("\n🔍 Top large files in Git history:")
    for i, (size, path) in enumerate(files):
        print(f"[{i}] {human_readable_size(size)} - {path}")

    indexes = input("\nEnter file numbers to remove (comma-separated): ")
    try:
        selected = [files[int(i.strip())] for i in indexes.split(",")]
        return selected
    except Exception as e:
        print("❌ Invalid input. Aborting.")
        exit(1)

def write_log_file(entries):
    os.makedirs("cleanup_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = f"logs/cleanup_log_{timestamp}.txt"

    with open(logfile, "w") as f:
        f.write("🧹 Git Cleanup Log\n")
        f.write(f"🕒 Time: {datetime.now()}\n\n")
        f.write("Files Removed:\n")
        for size, path in entries:
            f.write(f"- {path} ({human_readable_size(size)})\n")

    print(f"📝 Log saved to {logfile}")

def remove_files_from_history(filepaths):
    print("\n⚙️  Removing files from Git history...")
    paths = " ".join([f"--path '{path}'" for _, path in filepaths])
    cmd = f"git filter-repo {paths} --invert-paths --force"
    subprocess.run(cmd, shell=True)

def clean_git():
    print("🧽 Running Git garbage collection...")
    subprocess.run("rm -rf .git/refs/original/", shell=True)
    subprocess.run("git reflog expire --expire=now --all", shell=True)
    subprocess.run("git gc --prune=now --aggressive", shell=True)
    print("✅ Cleanup complete!")

def main():
    if not os.path.exists(".git"):
        print("❌ This is not a Git repository.")
        return

    files = get_largest_git_objects()
    if not files:
        print("✅ No large files found.")
        return

    to_remove = prompt_files_to_delete(files)
    if not to_remove:
        print("⚠️ No files selected.")
        return

    # Confirm
    dry = input("\n🧪 Run dry mode first? (y/n): ").lower()
    if dry == 'y':
        print("\n🔍 Dry run:")
        for size, path in to_remove:
            print(f"Would remove: {path} ({human_readable_size(size)})")
        if input("\nProceed with actual cleanup? (y/n): ").lower() != 'y':
            print("❌ Aborted.")
            return

    remove_files_from_history(to_remove)
    clean_git()
    write_log_file(to_remove)

if __name__ == "__main__":
    main()
