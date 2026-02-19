* SPICE netlist for the given schematic

VDD 3 0 DC <value_vdd>
Vin1 X 0 DC <value_vin1>
Vin2 Y 0 DC <value_vin2>

RD1 3 Vout <value_rd>
RD2 2 VDD <value_rd>

M1 Vout X 4 4 NMOS
M2 2 Y Vout Vout PMOS

Iss 4 0 DC <value_iss>

* NMOS and PMOS model parameters
.model NMOS nmos ( level=1 <other_parameters> )
.model PMOS pmos ( level=1 <other_parameters> )

* End of netlist