import sys, os
with open("/tmp/abaqus_debug.txt", "w") as f:
    f.write("cwd: %s\n" % os.getcwd())
    f.write("sys.argv: %s\n" % sys.argv)
    f.write("sys.path[:6]: %s\n" % sys.path[:6])
    f.write("abspath argv0: %s\n" % os.path.abspath(sys.argv[0]) if sys.argv else "no argv\n")
