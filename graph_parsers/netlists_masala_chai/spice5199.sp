spice
* Component declarations
V1 5 0 SINE(0 1 1k)
R1 3 2 1k
R2 2 4 10k
C1 4 7 10u
XU1 2 3 6 7 4 opamp
V+ 7 0 DC 15
V- 4 0 DC -15

* Subcircuit declaration for opamp
.subckt opamp +in -in out V+ V-
* Ideal opamp model
E1 out 0 VALUE { V(2,3) }
Rout out 0 1Meg
.ends opamp

* Analysis
.tran 1u 10m

* Control statements
.include 'standard.lib'
.end