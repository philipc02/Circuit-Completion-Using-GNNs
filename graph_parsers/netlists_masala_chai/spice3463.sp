spice
* SPICE netlist for the given circuit

* Voltage Sources
Va1 4 0 DC
Va2 2 0 DC

* NPN Transistors
Q1 1 4 8 NPN
Q2 7 2 9 NPN

* Resistors
Rb1 4 1 Rb
Rl1 5 6 Rl
Rc1 5 3 Rc
Rb2 2 7 Rb
Rl2 11 6 Rl
Rc2 5 3 Rc
R1a 8 10 R1
R1b 9 10 R1
2R1a 10 0 2R1
2R1b 10 0 2R1

* Capacitors
C1 4 5 C1
C2 2 6 C2

* Model Declarations
.model NPN NPN