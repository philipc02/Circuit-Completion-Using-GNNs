spice
* SPICE netlist for the given schematic

* Voltage Sources
VDD 2 0 DC <Value>
VGSL 3 0 DC <Value>
VGSD 6 0 DC <Value>
VI 1 0 DC <Value>

* PMOS Transistor
M1 2 3 4 4 PMOSModel

* NMOS Transistor
M2 4 6 0 0 NMOSModel

* Models
.model PMOSModel PMOS (Level=1 W=... L=...)
.model NMOSModel NMOS (Level=1 W=... L=...)

* Simulation commands
.tran 1n 100n
.end