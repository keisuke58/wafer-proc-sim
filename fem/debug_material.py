from abaqus import mdb, backwardCompatibility
backwardCompatibility.setValues(reportDeprecated=False)
from abaqusConstants import *
from caeModules import *

model = mdb.Model(name="debug")
m = model.Material(name="TestMat")
m.Elastic(table=((448e9, 0.21),))
m.Density(table=((3210.0,),))
m.DruckerPrager(shearCriterion=LINEAR, eccentricity=0.1, testData=OFF,
                temperatureDependency=OFF, dependencies=0,
                table=((68.8, 1.0, 0.0),))

with open("/tmp/abaqus_dp_debug.txt", "w") as f:
    f.write("DruckerPrager OK\n")
    dp = m.druckerPrager
    methods = sorted(a for a in dir(dp) if not a.startswith("_") and callable(getattr(dp, a)))
    f.write("druckerPrager methods: " + str(methods) + "\n")

    # Try hardening
    try:
        dp.DruckerPragerHardening(type=SHEAR, table=((1.0e9, 0.0),))
        f.write("DruckerPragerHardening(type=SHEAR, table=...) OK\n")
    except Exception as e:
        f.write("DruckerPragerHardening error: " + str(e) + "\n")
        # Try without type
        try:
            dp.DruckerPragerHardening(table=((1.0e9, 0.0),))
            f.write("DruckerPragerHardening(table=...) OK (no type)\n")
        except Exception as e2:
            f.write("Also failed: " + str(e2) + "\n")
