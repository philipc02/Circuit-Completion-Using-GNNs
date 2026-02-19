spice
* NMOS Transistor 1
M1 5 2 2 2 NMOS_Model W=width1 L=length1

* NMOS Transistor 2
M2 3 4 2 2 NMOS_Model W=width2 L=length2

* Current Source
I1 6 3 DC current_value

* Voltage Nodes
Vbias 2 0 DC Vbias_value
Vin 4 0 DC Vin_value

* Output
Vout 7 3 DC 0