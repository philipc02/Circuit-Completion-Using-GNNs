* SPICE netlist for the given circuit
* V_in connected between nodes 1 and 0
V_in 1 0 DC 0
* Rs connected between nodes 1 and 2
R_S 1 2 R_S_value
* Voltage source V1 connected between nodes 3 (positive) and 0 (negative)
V1 3 0 DC V1_value
* Current source controlled by V1 between nodes 3 and 5
GmValue 5 3 V1 gm_value
* Ro connected between nodes 5 and 7
Ro 5 7 Ro_value
* Capacitor Cl connected between 7 and 0
Cl 7 0 Cl_value
* Capacitor Cin connected between 2 and 0
Cin 2 0 Cin_value