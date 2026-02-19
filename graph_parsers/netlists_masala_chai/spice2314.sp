plaintext
* PMOS with drain (4), gate (2), source (3)
M1 4 2 3 3 PMOS

* Current Source I_S/2 connected between nodes 4 and 5
I1 4 5 DC I_S/2

* Voltage Source V_G connected between 6 and 2
V1 6 2 DC V_G

* Voltage Source V_DD connected to node 5
V2 5 0 DC V_DD

* Ground reference
V3 6 0 0