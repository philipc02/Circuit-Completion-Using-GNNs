* SPICE Netlist

VDD 1 0 DC 5V
V1 4 5 DC 1V
VGS 4 5 DC 2V

* Transistor models
.MODEL PMOS PMOS
.MODEL NMOS NMOS

* Transistors
ML 2 3 1 1 PMOS
MD 3 4 5 5 NMOS

* Nodes
* 1 - VDD
* 2 - Drain of ML, connected to VDD
* 3 - Source of ML, Drain of MD, output net VO
* 4 - Gate of MD, connected to V1 and VGS
* 5 - Source of MD, Ground

.END