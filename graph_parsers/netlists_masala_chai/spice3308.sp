spice
* List of components:
* M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11 (MOSFETs)
* VDD, Vcont (Voltage sources)
* Iss1, Iss2, Iss3 (Current sources)
* R1, R2 (Resistors)

* Voltage Sources
VDD 2 0 DC VDD
Vcont 6 0 DC Vcont

* Current Sources
Iss1 3 0 DC Iss1
Iss2 4 0 DC Iss2
Iss3 5 0 DC Iss3

* Resistors
R1 2 3 R1
R2 2 7 R2

* NMOS Transistors
M1 3 1 2 2 NMOS W=W L=L
M2 7 1 2 2 NMOS W=W L=L
M3 4 3 2 2 NMOS W=W L=L
M4 5 5 2 2 NMOS W=W L=L
M5 3 2 2 2 NMOS W=2W L=L
M6 2 1 2 2 NMOS W=2W L=L

* PMOS Transistors
M7 7 4 2 2 PMOS W=W L=L
M8 7 4 2 2 PMOS W=W L=L
M9 4 3 2 2 PMOS W=W L=L
M10 4 5 2 2 PMOS W=W L=L
M11 7 4 5 2 PMOS W=W L=L

* Connection Nodes
* Node 1: Vin
* Node 2: VDD
* Node 3: Output node at R1
* Node 4: Internal nodes
* Node 5: Internal nodes
* Node 6: Vcont
* Node 7: Vout 

.model NMOS NMOS
.model PMOS PMOS
.END