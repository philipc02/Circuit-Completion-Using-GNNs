plaintext
* SPICE Netlist for Amplifier Circuit

* Voltage Sources
VIC 3 0 DC V_ic
VID1 2 3 DC V_id/2
VID2 2 0 DC V_id/2
VDD 5 5 DC V_dd
VSS 4 0 DC -V_ss

* Operational Amplifier
* Connect input, output, and supply nodes
XOPAMP 2 2 VO 5 4 OPAMP_MODEL

* Output
VO 2 0 

* End of Netlist