spice
* SPICE Netlist for the Given Schematic

V1 1 2 DC V_s1
V2 5 6 DC V_s2

C1_1 1 4 C1
C1_2 5 4 C1
C2_1 2 4 C2
C2_2 2 4 C2
CL_1 2 0 CL
CL_2 3 0 CL

* Operational Amplifier
* Assume ideal with nodes connected as follows:
* Non-inverting input: Node 2
* Inverting input: Node 4
* Output: Node 2 and 3

* Operational amplifier connections
E_OPAMP 2 4 2 3 999MEG ; High gain op-amp assumption

.end