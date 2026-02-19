plaintext
*SPICE Netlist for the given schematic

*Voltage Sources
V1 VI 0 DC
VREF VREF 0 DC

*Resistors
RA VI 2 RA_VALUE
RB VREF 22 RB_VALUE
R1 2 0 R1_VALUE
R2 2 3 R2_VALUE

*Operational Amplifier
* (terminals: non-inv, inv, out)
XOP 2 22 VO OPAMP_MODEL

*End of netlist