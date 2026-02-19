plaintext
* SPICE netlist for the given schematic
* Node numbers are based on the annotated image

* Voltage Sources
V1 3 0 DC 10
V2 5 2 DC 5

* Current Source
I1 3 5 DC 0.5u

* Resistors
R1 3 4 10MEG
R2 5 4 10MEG
R3 2 6 6K
R4 2 7 6K

* NMOS Transistor
M1 6 5 7 7 NMOS

* .end statement to indicate end of netlist
.end