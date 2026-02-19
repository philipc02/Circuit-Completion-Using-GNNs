plaintext
* SPICE Netlist

* Current Sources
IB1 1 3 DC 0
IB2 4 3 DC 0

* Resistors
R1 2 5 1k
R2 5 3 1k

* Op-Amp (ideal model)
* Node 2: Inverting input
* Node 4: Non-inverting input
* Node 5: Output
XOPAMP 4 2 5 OPAMP

* Define the op-amp model
.subckt OPAMP non_inv inv out
VPLUS non_inv 0 DC 0
VOUT out 0 DC 0
G1 out 0 non_inv inv 1Meg
.ends OPAMP

.end