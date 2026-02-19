* SPICE Netlist
V1 1 0 DC <Voltage_Value1>
V2 4 0 DC <Voltage_Value2>

C1 1 2 1p
C2 2 3 1.5p
C3 3 5 0.5p
C4 4 2 1p
C5 2 6 1.5p
C6 6 5 0.5p

XOPAMP 2 4 1 3 OPAMP

* OPAMP model should be defined or included
* .include your_opamp_model.lib

.end