spice
* SPICE netlist for the given schematic
* V1 connected between nodes 6 and 0
V1 6 0 DC 

* Current sources
I1 6 7 DC
I3 3 22 DC
I4 23 0 DC

* NPN BJTs
* Q1: Base-8, Collector-3, Emitter-7
Q1 3 8 7 NPN

* Q2: Base-3, Collector-4, Emitter-0
Q2 3 4 0 NPN

* Q3: Base-3, Collector-2, Emitter-0
Q3 3 2 0 NPN

* Resistors
* R1 connected between nodes 8 and 3
R1 8 3 R_R1

* R2 connected between nodes 3 and 4
R2 3 4 R_R2

* R3 connected between nodes 3 and 2
R3 3 2 R_R3

* Capacitor
* C1 connected between nodes 8 and 0
C1 8 0 21u

* Load resistor related to i3
* Assuming i3 to be related to a load, if applicable
RL 2 1 R_L

.END