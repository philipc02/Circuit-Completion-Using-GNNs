plaintext
* Differential Amplifier SPICE Netlist

VCC 7 0 DC [value] ; replace [value] with the actual voltage

* Transistors
Q1 3 2 4 NPN
Q2 3 5 6 NPN

* Resistors
R1 6 7 [value] ; replace [value] with actual resistance
R2 16 15 [value]
R3 3 13 [value]
R4 5 17 [value]
R5 19 7 [value]
R6 4 18 [value]

* Capacitors
C1 vi 10 [value] ; replace [value] with actual capacitance
C2 18 11 [value]
C3 19 vo [value]

* Voltage Sources
vi 10 0 AC 1 ; input voltage source