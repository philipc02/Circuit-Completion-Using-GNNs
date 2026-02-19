plaintext
* SPICE Netlist

VDD 4 0 DC VDD

* NMOS Transistor
* M<name> <drain> <gate> <source> <body> <model> [L=<length>] [W=<width>]
M1 4 2 1 1 NMOS

* PMOS Transistor
* M<name> <drain> <gate> <source> <body> <model> [L=<length>] [W=<width>]
M2 2 2 3 3 PMOS

* Capacitor
* C<name> <node1> <node2> <value>
C1 2 5 C

* Model Definitions
.model NMOS NMOS (level=1)
.model PMOS PMOS (level=1)

.end