spice
* SPICE Netlist

* Diode with series current source
D1 2 5 DMOD
I_IS 5 2 DC I_S

* Diode with series current source
D2 4 2 DMOD
I_ISR 44 4 DC I_SR

* Diode with series current source
D3 3 4 DMOD
I_IKP 0 3 DC I_KP

* Resistor
RS 3 0 RS_VALUE

* Models
.model DMOD D