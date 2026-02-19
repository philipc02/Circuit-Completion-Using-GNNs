spice
* Components
MN1 4 3 7 7 NMOS
MN2 4 1 5 5 NMOS
R1 4 5 R1
C1 2 5 C1
V1 1 0 DC

* Nodes
* 1 - Input (Vi)
* 2 - Output (Vo)
* 3 - Gate of MN2
* 4 - Drain of MN1 and MN2, connected to R1
* 5 - Source of MN2, connected to ground
* 7 - Source of MN1, connected to ground

* Values
.param R1 = A0*RL
.param C1 = CL