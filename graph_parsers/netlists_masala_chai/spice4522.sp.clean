* SPICE Netlist for the given schematic

R1 1 2 4.4k
R2 2 0 10k
R3 2 3 10k
D1 0 2 D_ZENER
* Model required: .model D_ZENER D(...parameters...)

* Assuming op-amp as a generic model
XU1 2 0 3 5 OPAMP
* OPAMP: Non-inverting input (2), Inverting input (0), Output (3), Power rail connections

* Sources
* V+ is assumed as a DC voltage source if needed; define it accordingly

* Zener Voltage Vz (if known)
* .DC V+ ... (Specify value if needed)
* .MODEL OPAMP ... (specify opamp parameters)

.end