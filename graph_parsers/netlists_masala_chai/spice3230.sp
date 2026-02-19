plaintext
* SPICE Netlist

* Voltage Source
V1 4 0 αVin

* Capacitors
Ceq 4 2 Ceq
C2 3 2 C2
CL 3 0 CL

* Voltage-Controlled Current Source
G1 3 0 VALUE={Gm*V(2,0)}

* Resistor
R0 3 5 R0