plaintext
* Transistor Definitions (Assuming NPN BJT)
Q1 V1 Vb1 net1 QNPN
Q2 net1 Vb2 0 QNPN

* Resistor
RC VCC V1 1k

* Voltage Source
VCC VCC 0 DC 10V

* Model Definitions
.model QNPN NPN (IS=1E-14 BF=100)

* End of netlist