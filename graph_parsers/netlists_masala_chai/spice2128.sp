* SPICE Netlist
V1 3 0 DC <value>    ; Voltage source V1 connected to net 3 and ground
VX 2 0 DC <value>    ; Voltage source Vx connected to net 2 and ground
RS 3 4 1k            ; Resistor Rs connected between net 3 and net 4
GM 4 3 VCR 0 4 1    ; Controlled current source gm*v1 between current measurement node 4 and 3
RO 3 2 1k            ; Resistor ro connected between net 3 and net 2

.END