plaintext
* SPICE Netlist for the given circuit

* PMOS Transistors
MMP1 2 6 10 10 PMOS
MMP2 9 2 10 10 PMOS

* NMOS Transistors
MMNA 2 3 4 4 NMOS
MMNB 3 4 5 5 NMOS
MMN1 4 5 0 0 NMOS

* Voltage Supply
VDD 10 0 DC VDD

* Inputs
VIN_CLK 8 0 PULSE(0 VDD 0 1n 1n 100n 200n)
VIN_A 3 0 DC 0
VIN_B 4 0 DC 0

* Signal Outputs
* VOUT 9

* End of netlist