plaintext
* SPICE Netlist
Q1 5 2 6 NPN
Q2 9 2 3 NPN
Q3 5 8 2 PNP
Q4 8 5 10 PNP
Q5 8 3 7 PNP
Q6 9 5 2 NPN

R2 7 8 <Value> ; Replace <Value> with the actual resistance value

IBIAS1 6 2 DC <Value> ; Replace <Value> with the actual current value
IBIAS2 4 10 DC <Value> ; Replace <Value> with the actual current value

VPLUS 8 0 DC <Voltage> ; Replace <Voltage> with the actual voltage value for +V
VMINUS 4 0 DC <Voltage> ; Replace <Voltage> with the actual voltage value for -V

* Note: Replace <Value> and <Voltage> with actual values
.END