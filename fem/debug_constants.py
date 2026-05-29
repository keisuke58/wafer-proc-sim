from abaqusConstants import *
from caeModules import *
import abaqusConstants as ac

with open("/tmp/abaqus_constants_debug.txt", "w") as f:
    checks = ["MAXPS", "MAXPE", "MAX_PRINCIPAL_STRESS", "QUADS",
              "DISCRETE_RIGID_SURFACE", "ANALYTICAL_RIGID_SURFACE",
              "DRUCKER_PRAGER", "JOHNSON_COOK", "DISTORTION_CORRECTION"]
    for c in checks:
        f.write("%s: %s\n" % (c, c in dir(ac) or c in dir()))

    # Also check Part type constants
    f.write("\n-- Part types --\n")
    part_types = sorted(n for n in dir(ac) if "RIGID" in n or "DEFORM" in n)
    f.write("\n".join(part_types))
    f.write("\n\n-- Damage criteria --\n")
    crit = sorted(n for n in dir(ac) if any(x in n for x in ["MAXP","MAXS","MAXE","QUADS","QUADE","HASHIN","PUCK"]))
    f.write("\n".join(crit))
