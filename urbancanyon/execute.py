import subprocess

def execute_main():
    print("EXE: start running raytracer script")
    subprocess.call("sh runme", shell=True)
    print("EXE: finished running raytracer script")
    return

if __name__ == "__main__":
    execute_main()
