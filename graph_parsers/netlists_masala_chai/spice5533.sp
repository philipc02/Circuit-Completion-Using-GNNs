plaintext
* SPICE Netlist for the given circuit

* Nodes:
* 1 - current source input
* 2 - node on the left of Ri, VCVS, and Ro connections
* 3 - node between Ri and ground
* 4 - node on the right of Ro and Io node
* 5 - ground 

* Current Source
I1 1 2 DC IIN

* Resistors
R1 2 3 RI
R2 2 4 RO

* Voltage-Controlled Voltage Source
E1 2 3 2 5 RM

* Ground
V0 3 5 DC 0