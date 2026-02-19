plaintext
* Zener Diode Voltage Regulator Circuit

V1 4 0 DC <Voltage_Value>
RS 4 5 <Resistance_Value_Rs>
R2 5 3 <Resistance_Value_R2>
ZD1 5 2 D1

.model D1 D(IS=1e-14 N=1)

.END