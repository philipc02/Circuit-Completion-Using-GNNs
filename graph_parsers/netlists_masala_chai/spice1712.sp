plaintext
* SPICE Netlist

* Voltage Source
Vin 1 0 DC 0

* Resistors
Rs 1 3 200
RD1 5 2 1k
Rf 3 X 10k
RD2 7 8 1k

* MOSFETs (Assuming default model parameters)
M1 5 3 4 4 NMOS
M2 X 6 2 2 NMOS

* Analysis
.ac dec 10 1k 1Meg

* End of netlist