plaintext
* SPICE Netlist for the CMOS circuit

* Transistors
M2 N002 VDD VOUT VOUT PMOS L=1u W=1u
M1 N002 Vb VSS VSS NMOS L=1u W=1u
M3 N003 N002 VSS VSS NMOS L=1u W=1u

* Voltage Source
VDD VDD 0 DC 5V

* .model statement for MOSFETs assuming generic 180nm process
.model NMOS NMOS(Level=1 Vto=0.7)
.model PMOS PMOS(Level=1 Vto=-0.7)

* Simulation commands
*.tran 1n 100n
*.end