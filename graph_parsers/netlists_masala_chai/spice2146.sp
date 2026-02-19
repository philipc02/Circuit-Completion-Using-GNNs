spice
* SPICE Netlist
VDD 5 0 DC <Voltage Value>

RG Vin 3 <RG_Value>
R1 5 3 <R1_Value>
R2 3 0 <R2_Value>
RD 5 4 <RD_Value>
RS 2 0 <RS_Value>

C1 3 0 <C1_Value>
C2 2 0 <C2_Value>

M1 4 3 2 2 NMOS

* Control Statements
.control
run
.endc