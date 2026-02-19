spice
* SPICE Netlist

* NMOS Transistors
M1 2 1 4 4 NMOS
M2 4 3 7 7 NMOS
M3 5 3 7 7 NMOS

* Current Source
IREF 6 5 DC 100uA

* Resistor
R 6 5 Rvalue

* Specify further parameters or models if necessary
.MODEL NMOS NMOS (LEVEL=1)

* Nodes
* 1 - vi (input)
* 2 - Vdd (positive voltage supply)
* 3 - Connection between gates of M2 and M3
* 4 - vo (output)
* 5 - Node connected to the bottom of the resistor and top of the M3
* 6 - Top of the current source and resistor
* 7 - Ground (common ground)