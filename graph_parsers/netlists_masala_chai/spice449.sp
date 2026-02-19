plaintext
* Transistor NMOS: Drain Gate Source
M1 4 1 3 NMOS
M2 2 3 4 NMOS
M3 2 7 2 NMOS
M4 3 2 2 NMOS
M9 6 5 2 NMOS

* Transistor PMOS: Drain Gate Source
M5 4 2 8 PMOS
M6 8 5 2 PMOS
M7 6 4 8 PMOS
M8 4 1 8 PMOS

* Current Source
I1 7 2 DC 20uA

* Capacitors
C1 6 5 5pF

* Resistors as load (100k / 50k)
R1 4 8 100k
R2 6 8 100k
R3 2 3 50k
R4 3 2 50k

* Voltage Sources
V1 1 2 DC +5V
V2 2 7 DC -5V

* .MODEL definitions (assuming standard models)
.model NMOS NMOS
.model PMOS PMOS

* End of netlist