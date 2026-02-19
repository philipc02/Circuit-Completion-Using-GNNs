plaintext
* PMOS Transistors
M1 2 2 4 4 PMOS
M2 5 4 3 3 PMOS

* NMOS Transistors
M3 3 2 6 0 NMOS
M4 3 5 0 0 NMOS

* Voltage source defined at node 2
VDD 2 0 DC VDD

* Nodes:
* 0 - Ground
* 2 - VDD, Drain of QPA, Gate of QPB
* 3 - Source of QNA, Drain of QNA, QNB, Output Y
* 4 - Source of QPA, Gate of QNA, Drain of QPB
* 5 - Source of QPB, Gate of QNB
* 6 - Source of QNB
*
* End of netlist