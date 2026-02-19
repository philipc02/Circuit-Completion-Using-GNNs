spice
* SPICE Netlist for the given schematic

* Voltage Sources
VY VY 0 DC <Vy_value>
VN VN 0 DC <Vn_value>
VX VX 0 DC <Vx_value>

* Resistors
R3 VY 2 R3_value
R4 VN 2 R4_value
R5 VX 4 R5_value
R6 2 Vout R6_value

* Operational Amplifier (Ideal)
* Assuming ideal op-amp model, connected as an inverting amplifier
* Input at Node 2, reference at Node 4, output at Vout
* This is a conceptual representation for ideal op-amp behavior
*

* End of netlist