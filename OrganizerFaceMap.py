import os
import subprocess
import json
import datetime

LOG_FILE = os.path.expanduser("~/.organizer_facemap_log.json")

def organize_files(root_dir):
    find_command = ['find', root_dir, '-type', 'f', '(', '-iname', '*_FacemapPose.h5', '-o', '-iname', '*_FacemapPose_metadata.pkl', ')']
    result = subprocess.run(find_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    files = [f for f in result.stdout.strip().split('\n') if f]

    find_avi = ['find', root_dir, '-type', 'f', '-iname', '*.avi']
    result_avi = subprocess.run(find_avi, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    avi_files = [f for f in result_avi.stdout.strip().split('\n') if f]

    prefix_to_dir = {}
    for avi in avi_files:
        basename = os.path.basename(avi) 
        # Dictionary mapping video base name to its directory
        # Used to move .h5 files next to their corresponding .avi file
        prefix = basename.replace('.avi', '')
        prefix_to_dir[prefix] = os.path.dirname(avi)

    success = True
    
    # Track number of successful moves and log entries for undo functionality
    moved = 0 #count number of move 
    log_entries = [] # save the moving logs to undo the organization if needed

    for f in files:
        basename = os.path.basename(f)
        prefix = basename.split('_FacemapPose')[0] #loops on every "_FacemapPose" = every .h5 and metadata.pkl
        if prefix in prefix_to_dir:
            target_dir = prefix_to_dir[prefix] #find de .avi that directs the moving
            target_path = os.path.join(target_dir, basename) #creates the path to the .avi 
            # Only move if file is not already in the correct location
            if os.path.abspath(f) != os.path.abspath(target_path):
                subprocess.run(['mv', f, target_path])
                print(f"Moved: {basename} -> {target_dir}")
                log_entries.append({"source": f, "destination": target_path})
                moved += 1
            else:
                print(f"Already in place: {basename}")
        else:
            print(f"WARNING: No matching .avi found for {basename}")
            success = False

    if log_entries: #writes log only if there's a movement 
        run_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "root_dir": root_dir,
            "moves": log_entries
        }
        history = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        # Append this run to the log history (never overwrites previous runs)
        # Allows undoing any organization with undo_organization()
        history.append(run_log)
        with open(LOG_FILE, 'w') as f:
            json.dump(history, f, indent=2)

    if success:
        print(f"\nOrganized successfully — {moved} file(s) moved.") #\n makes an empty line in the terminal to make the message more visible
    else:
        print(f"\nOrganisation failed — some files could not be matched.")


def undo_last_run():
    """Undo the last organize_files() run by reversing all file moves.
   Uses the log file at LOG_FILE to find the last run.
   Pops the last run from history so it cannot be undone twice.
   """
    if not os.path.exists(LOG_FILE):
        print("No history found — nothing to undo.")
        return

    with open(LOG_FILE, 'r') as f: #search the file with all the log of the movements
        history = json.load(f)

    if not history:
        print("No history to undo.")
        return

    last_run = history.pop() #take the last element of the lists of runs to undo only the last organisation
    print(f"Undoing run from {last_run['timestamp']} ({len(last_run['moves'])} files)...")

    for move in reversed(last_run['moves']): #source and destination are inversed
        src = move['destination']
        dst = move['source']
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            subprocess.run(['mv', src, dst])
            print(f"Restored: {os.path.basename(src)} -> {dst}")
        else:
            print(f"WARNING: {src} not found — cannot undo this move.")

    with open(LOG_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print("\nUndo complete.")


if __name__ == "__main__":
    mode = input("Type 'undo' to cancel the last storing and 'Enter' to launch a new storing : ").strip()
    if mode.lower() == "undo":
        undo_last_run()
    else:
        path = input("Common path to the files containing the video : ")
        organize_files(path)
