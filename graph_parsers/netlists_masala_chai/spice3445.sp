plaintext
* SPICE Netlist for given circuit

VVT 4 6 DC <value_of_VT>
VVDD 1 0 DC <value_of_VDD>
VVSS 8 0 DC <value_of_VSS>

RT 6 2 <value_of_RT>
RD 1 2 <value_of_RD>
RS 2 5 <value_of_RS>
RL 2 3 <value_of_RL>

CS 5 8 <value_of_CS>

M1 2 6 5 5 NMOS

* Define the model for NMOS
.model NMOS NMOS (level=1)

.end