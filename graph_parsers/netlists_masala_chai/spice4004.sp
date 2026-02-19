spice
* SPICE netlist for the provided circuit

* Current source
I1 3 2 DC Iq

* Resistor RS
RS 1 2 RS_value

* Resistor Ro
Ro 4 0 Ro_value

* NPN Transistor
Q1 3 1 0 QNPN

* Analysis (example for DC analysis)
.DC Vi 0 5 0.1

* Models and parameters
.model QNPN NPN (IS=1e-14 BF=100)

* Simulation commands
.end