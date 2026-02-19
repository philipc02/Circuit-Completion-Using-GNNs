spice
* SPICE Netlist for the given schematic
* Nodes are identified based on the annotated image.

* Voltage Input (Node Vi)
Vi 1 0 DC

* Input Capacitor (Ideal, behaves as an open circuit at DC)
C1 1 2 1e9

* 3MΩ Resistor
R1 2 3 3e6

* NMOS Transistor - Assume default model (M1 is NMOS)
* Drain Gate Source
M1 4 2 5 5 NMOS

* Capacitor across nodes 3 and 4
C2 3 4 1e9

* 2MΩ Resistor
R2 4 0 2e6 

* Current Source 200 μA
I1 3 7 DC 200u

* Output Capacitor (Ideal, behaves as an open circuit at DC)
Cout 7 6 1e9

* Output node
Vo 6 0

* .model NMOS device
.model NMOS NMOS