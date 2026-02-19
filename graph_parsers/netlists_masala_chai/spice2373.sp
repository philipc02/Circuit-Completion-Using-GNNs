plaintext
* SPICE Netlist

* Voltage Sources
V1 1 0 DC VDD
V2 3 0 DC VSS

* Current Source
I1 1 6 DC 2.5uA

* PMOS Transistors
M1 2 3 1 1 PMOS L=0.2u W=125u
M2 2 3 2 2 PMOS L=0.2u W=125u

* NMOS Transistors
M3 4 5 3 3 NMOS L=0.5u W=20u
M4 5 3 3 3 NMOS L=0.5u W=20u
M5 5 5 4 4 NMOS L=0.5u W=20u
M6 2 0 3 3 NMOS L=0.5u W=20u
M7 3 0 3 3 NMOS L=0.5u W=20u

* Connections
* Node 2: V_IN21
* Node 3: V_OUT21
* Node 4: V_OUT22
* Node 5: Gate Connection for NMOS
* Node 6: V_IN22