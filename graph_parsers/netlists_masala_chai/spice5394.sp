* Op-Amp Based Filter Circuit

* Resistor between output and inverting input
Rf 3 4 1k

* Capacitors and Resistors in series after the op-amp output
C1 4 2 0.001u
R1 2 0 15k

C2 2 1 0.001u
R2 1 0 15k

C3 1 6 0.001u
R3 6 0 15k

* Op-Amp
* Ideal op-amp model needs to be defined separately in SPICE or use built-in SPICE op-amp model
XOP 5 3 6 opamp

* Node Assignments
* 1 - Node at first capacitor and resistor series
* 2 - Node at second capacitor and resistor series
* 3 - Common input node
* 4 - Rf and op-amp inverting input node
* 5 - Non-inverting input (ground)
* 6 - Output node
* 8 - Vout

* End of netlist