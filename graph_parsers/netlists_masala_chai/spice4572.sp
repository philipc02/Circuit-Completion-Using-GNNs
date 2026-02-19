spice
* NMOS Transistors
MX 4 v_X 0 NMOS_MODEL
MY 3 v_Y 0 NMOS_MODEL
MZ 3 v_Z 0 NMOS_MODEL

* Voltage Source
VDD 5 0 DC 3V
VTNL 5 2 DC -1V

* Bipolar Transistor (assuming NPN)
QL 5 2 0 NPN_MODEL

* Netlist ends
.end