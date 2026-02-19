spice
* Components
R1 1 3 100k
R2 3 4 100k
R3 4 2 100k
R4 6 5 100k
V1 1 0 v1
V2 6 0 v2

* Operational amplifier
* Node 3 is non-inverting input (A), Node 4 is inverting input (B)
XOP 4 3 2 opamp

* Ground node
VSS 5 0 0

* End of netlist
.end