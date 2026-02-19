spice
* SPICE netlist for the given circuit

VDD 3 0 DC <VDD_value>           ; Supply voltage
Vin Vin 0 DC <Vin_value>         ; Input voltage

* NMOS Transistors
M1 X Vin 0 0 NMOS                ; M1: Drain=X, Gate=Vin, Source=0
M2 4 Vb X X NMOS                 ; M2: Drain=4, Gate=Vb, Source=X

* Resistors
RD1 3 X <RD1_value>              ; RD1: Connected between VDD and X
RD2 4 Vout <RD2_value>           ; RD2: Connected between VDD and Vout

* .MODEL statements for NMOS
.model NMOS NMOS(Level=1 VTO=0.7 KP=20u)

.end